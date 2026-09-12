'''
NEMO curvilinear-grid reader.

Reads ssh/ubar/vbar, unrotates ubar/vbar from the model's own grid-relative
(i,j) axes to true east/north, colocates everything onto T-points, and NaNs
land cells. Output is ready for the regridder.py / grid_interp.py pipeline.

Also holds `read_nemo_bathymetry`, for the static `bathy_metry` field in the
same domain_cfg.nc used above for glamt/gphit/glamv/gphiv.
'''

from pathlib import Path

import numpy as np
import xarray as xr

# leftover source-file coords that don't apply to the merged output
_SOURCE_COORDS_TO_DROP = ["nav_lat", "nav_lon", "time_centered"]


def _bearing(lon1, lat1, lon2, lat2):
    ''' Great-circle bearing from (lon1,lat1) to (lon2,lat2), radians clockwise
    from true north. Inputs in degrees; vectorized. '''
    lon1, lat1, lon2, lat2 = (np.deg2rad(a) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def compute_t_point_angle(domain_cfg):
    '''
    Grid rotation angle at each T-point: (gcost, gsint), the cos/sin of the
    local j-axis's compass bearing, from domain_cfg's glamv/gphiv neighbors.

    NOTE: domain_cfg doesn't store this angle directly, so we derive it the
    same way NEMO's own geo2ocean.F90::angle does, from the same coordinates.

    Returns
    -------
    gcost, gsint : np.ndarray, shape (y, x)
    '''
    glamv, gphiv = domain_cfg["glamv"].values, domain_cfg["gphiv"].values
    glamt, gphit = domain_cfg["glamt"].values, domain_cfg["gphit"].values

    lon_south = np.roll(glamv, 1, axis=0)
    lat_south = np.roll(gphiv, 1, axis=0)
    lon_south[0, :] = glamt[0, :]  # one-sided at j=0: no south V neighbor
    lat_south[0, :] = gphit[0, :]

    theta_j = _bearing(lon_south, lat_south, glamv, gphiv)
    return np.cos(theta_j), np.sin(theta_j)


def _to_t_point(u, v):
    '''
    Colocate U-point `u` and V-point `v` onto T-points by averaging each
    pair of neighbors straddling it (U(i-1,j)/U(i,j) in x, V(i,j-1)/V(i,j)
    in y). Falls back to the single neighbor at the i=0/j=0 edge.

    NOTE: skips a NaN (land) neighbor instead of averaging it in -- a plain
    mean would bleed NaN one cell inland from every coastline.

    Returns
    -------
    u_t, v_t : xr.DataArray
    '''
    u_west = u.shift(x=1).fillna(u)  # one-sided at x=0: no west neighbor
    u_t = xr.concat([u_west, u], dim="_pair").mean("_pair", skipna=True)

    v_south = v.shift(y=1).fillna(v)  # one-sided at y=0: no south neighbor
    v_t = xr.concat([v_south, v], dim="_pair").mean("_pair", skipna=True)

    return u_t, v_t


def unrotate_to_geographic(u_i, v_j, gcost, gsint):
    '''
    Convert NEMO's grid-relative (i,j) vector components to true (east,north).

    NOTE: derived from the compass-bearing definition of the local j-axis
    (j_hat = (gsint, gcost) in East/North) plus the grid's i/j orthogonality
    (validated against the real domain_cfg to within 0.06 deg). This is the
    transpose of grid_interp._rotate_vectors, as expected since that function
    rotates the opposite direction (true -> local). Re-derived and verified
    independently rather than trusted from a web-fetched paraphrase of NEMO's
    geo2ocean.F90 source, which had the sign on gsint backwards.
    '''
    u_east = u_i * gcost + v_j * gsint
    v_north = v_j * gcost - u_i * gsint
    return u_east, v_north


class NemoOceanReader:
    '''
    Constructed once by Hydra.

    Reads one row of NEMO ocean files (ssh and/or ubar/vbar; see
    domain.file_match.nemo_ocean) and returns one merged dataset, colocated
    onto domain_cfg's T-point grid. ubar/vbar are unrotated to true
    east/north. Matches the read_nc/read_tif reader_fn contract.

    Any subset of {ssh, ubar, vbar} works -- e.g. `[ssh]` alone, or
    `[ubar, vbar]` alone -- just keep dataset.variable_names/pair_vars_list
    in sync with whatever subset is active.

    NOTE: files are matched to a variable by name, not position, so
    file_match's token order doesn't matter. ubar/vbar must be given
    together (or not at all), since rotating one without the other isn't
    meaningful.

    Parameters
    ----------
    domain_cfg_path : str or Path
        Path to the NEMO domain_cfg.nc providing glamt/gphit/glamv/gphiv.
    ssh_var, u_var, v_var : str
        Variable names as they appear in the source files.
    '''

    def __init__(self, domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar"):
        self.ssh_var = ssh_var
        self.u_var = u_var
        self.v_var = v_var

        with xr.open_dataset(domain_cfg_path) as domain_cfg:
            gcost, gsint = compute_t_point_angle(domain_cfg)
            self._gcost = xr.DataArray(gcost, dims=("y", "x"))
            self._gsint = xr.DataArray(gsint, dims=("y", "x"))
            self._lat = domain_cfg["gphit"].values
            self._lon = domain_cfg["glamt"].values

            tmask = domain_cfg["top_level"].values > 0  # True = ocean, at T-points
            self._ocean_mask = xr.DataArray(tmask, dims=("y", "x"))

            # U(i,j)/V(i,j) are wet only if both T-cells straddling them are wet
            u_mask = tmask & np.roll(tmask, -1, axis=1)
            u_mask[:, -1] = False
            v_mask = tmask & np.roll(tmask, -1, axis=0)
            v_mask[-1, :] = False
            self._u_mask = xr.DataArray(u_mask, dims=("y", "x"))
            self._v_mask = xr.DataArray(v_mask, dims=("y", "x"))

    def _match_files(self, files):
        ''' map each of ssh_var/u_var/v_var to whichever file's name contains it '''
        file_by_var = {}
        for var in (self.ssh_var, self.u_var, self.v_var):
            matches = [f for f in files if var in Path(f).stem]
            if len(matches) > 1:
                raise ValueError(f"multiple files match variable {var!r}: {matches}")
            if matches:
                file_by_var[var] = matches[0]

        unmatched = set(files) - set(file_by_var.values())
        if unmatched:
            raise ValueError(
                f"file(s) don't match any of ssh_var/u_var/v_var "
                f"({self.ssh_var!r}/{self.u_var!r}/{self.v_var!r}): {sorted(unmatched)}"
            )
        return file_by_var

    def __call__(self, files):
        ''' read, unrotate, colocate, and mask -- returns [merged_dataset] '''
        file_by_var = self._match_files(files)
        has_u, has_v = self.u_var in file_by_var, self.v_var in file_by_var
        if has_u != has_v:
            raise ValueError(
                f"{self.u_var}/{self.v_var} must both be given to rotate to true "
                f"east/north -- got only {sorted(file_by_var)}"
            )

        opened = {var: xr.open_dataset(path, chunks={}) for var, path in file_by_var.items()}
        time_counters = [ds["time_counter"] for ds in opened.values()]
        assert all(tc.equals(time_counters[0]) for tc in time_counters[1:]), \
            f"time_counter mismatch across files: {files}"

        data_vars = {}
        for var, ds in opened.items():
            da = ds[var].drop_vars(_SOURCE_COORDS_TO_DROP, errors="ignore")
            if da.shape[-2:] != self._lat.shape:
                raise ValueError(
                    f"{file_by_var[var]}: shape {da.shape[-2:]} doesn't match "
                    f"domain_cfg's T-grid {self._lat.shape}"
                )
            data_vars[var] = da

        if has_u and has_v:
            # mask land before colocating, so it can't leak into a coastal average
            u_da = data_vars[self.u_var].where(self._u_mask)
            v_da = data_vars[self.v_var].where(self._v_mask)
            u_t, v_t = _to_t_point(u_da, v_da)
            data_vars[self.u_var], data_vars[self.v_var] = unrotate_to_geographic(
                u_t, v_t, self._gcost, self._gsint
            )

        merged = xr.Dataset(
            data_vars,
            coords={
                "lat": (("y", "x"), self._lat),
                "lon": (("y", "x"), self._lon),
            },
        )
        merged = merged.rename({"time_counter": "time"})

        # NaN land T-point
        merged = merged.where(self._ocean_mask)

        return [merged]


def read_nemo_bathymetry(files, variable_name="bathy_metry"):
    '''
    Read static bathymetry from a NEMO domain_cfg.nc's T-point `bathy_metry`,
    masking land (`top_level == 0`) to NaN. Matches the read_nc/read_tif
    reader_fn contract (`dataset.static: true`, so this is called once).

    Parameters
    ----------
    files : list of one path
        The domain_cfg.nc file (see domain.file_match's "domain_cfg" token).
    variable_name : str
        Bathymetry variable name in domain_cfg.

    Returns
    -------
    list of one xr.Dataset
    '''
    [file] = files
    with xr.open_dataset(file) as domain_cfg:
        depth = domain_cfg[variable_name].where(domain_cfg["top_level"] > 0)
        ds = xr.Dataset(
            {variable_name: (("y", "x"), depth.values)},
            coords={
                "lat": (("y", "x"), domain_cfg["gphit"].values),
                "lon": (("y", "x"), domain_cfg["glamt"].values),
            },
        )
    return [ds]
