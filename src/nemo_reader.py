'''
---> This script is specific to reading a target dataset <---
NEMO curvilinear-grid reader:
    1. unrotates `ubar`/`vbar` from the model's own grid-relative (i,j) axes to true east/north
    2. colocates `ssh`/`ubar`/`vbar` onto a common T-point grid
    3. NaNs land cells using `domain_cfg`'s `top_level` T-point mask

Now, data is ready to pass to the `regridder.py` / `grid_interp.py` pipeline, which expects vector
variables already expressed in true east/north.
'''

import numpy as np
import xarray as xr

# non-dimension coordinates carried over from the source files that don't apply to the merged
_SOURCE_COORDS_TO_DROP = ["nav_lat", "nav_lon", "time_centered"]


def _bearing(lon1, lat1, lon2, lat2):
    '''
    Great-circle initial bearing from (lon1,lat1) to (lon2,lat2), in radians,
    clockwise from true north (0 = north, pi/2 = east). Inputs in degrees.
    Vectorized (numpy broadcasting).
    '''
    lon1, lat1, lon2, lat2 = (np.deg2rad(a) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def compute_t_point_angle(domain_cfg):
    '''
    Derive NEMO's grid rotation angle at T-points from `domain_cfg`: (gcost, gsint),
    the cos/sin of the local grid j-axis's compass bearing (from true north), from
    `glamv`/`gphiv` (V-point) neighbors straddling each T-point in the j-direction
    (V(j-1,i) south, V(j,i) north); one-sided (single neighbor) at j=0. This is the
    same coordinate-derived approach NEMO's own `geo2ocean.F90::angle` uses, since
    the angle isn't stored in `domain_cfg` directly.

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
    Colocate NEMO's U-point `u` and V-point `v` (dims (..., "y", "x")) onto
    T-points via NaN-aware pairwise averaging along the staggering axis
    (T(i,j) straddled by U(i-1,j)/U(i,j) in x, and V(i,j-1)/V(i,j) in y) --
    ignoring whichever neighbor is NaN (land) instead of propagating it, which
    would otherwise bleed NaN one cell inland from every coastline. One-sided
    (falls back to the single available neighbor) at the i=0/j=0 edge. Stays
    lazy/dask-friendly (no `.values`/eager computation).

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

    Derivation: by definition of compass bearing theta_j (clockwise from true
    north), the local +j-axis unit vector is j_hat = (sin theta_j, cos theta_j)
    = (gsint, gcost) in (East, North). The grid is orthogonal with the +i axis
    90 deg clockwise from +j (validated against the real domain_cfg: local
    grid-i vs grid-j bearings differ by 90 deg to within 0.06 deg std across
    ~48k ocean points), so i_hat = (sin(theta_j+90), cos(theta_j+90)) =
    (gcost, -gsint). A vector with grid components (u_i, v_j) is
    u_i*i_hat + v_j*j_hat, giving:
        u_east  =  u_i*gcost + v_j*gsint
        v_north = -u_i*gsint + v_j*gcost
    This is the matrix transpose of `grid_interp._rotate_vectors` (expected,
    since that function rotates the opposite direction: true -> local basis).
    (Re-derived and numerically verified independently rather than trusting a
    web-fetched paraphrase of NEMO's `geo2ocean.F90` source for the sign.)
    '''
    u_east = u_i * gcost + v_j * gsint
    v_north = v_j * gcost - u_i * gsint
    return u_east, v_north


class NemoOceanReader:
    '''
    Constructed once by Hydra

    Reads one row of paired NEMO ocean files:
    `ssh` (T-point), `ubar`, `vbar` (U/V-point); see `domain.file_match: nemo_ocean`
    and returns a single merged dataset with all three variables colocated onto
    `domain_cfg`'s T-point grid.
    
    `ubar`/`vbar` unrotated from grid-relative to true east/north.
    Matches the `read_nc`/`read_tif` reader_fn contract.

    Parameters
    ----------
    domain_cfg_path : str or Path
        Path to the NEMO `domain_cfg.nc` (or subset thereof) providing
        `glamt`/`gphit`/`glamv`/`gphiv`.
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

    def __call__(self, files):
        ssh_file, u_file, v_file = files

        ssh_ds = xr.open_dataset(ssh_file, chunks={})
        u_ds = xr.open_dataset(u_file, chunks={})
        v_ds = xr.open_dataset(v_file, chunks={})

        assert u_ds["time_counter"].equals(ssh_ds["time_counter"]) and \
            v_ds["time_counter"].equals(ssh_ds["time_counter"]), \
            f"time_counter mismatch across ssh/ubar/vbar files: {files}"

        ssh_da = ssh_ds[self.ssh_var].drop_vars(_SOURCE_COORDS_TO_DROP, errors="ignore")
        u_da = u_ds[self.u_var].drop_vars(_SOURCE_COORDS_TO_DROP, errors="ignore")
        v_da = v_ds[self.v_var].drop_vars(_SOURCE_COORDS_TO_DROP, errors="ignore")

        if ssh_da.shape[-2:] != self._lat.shape:
            raise ValueError(
                f"{ssh_file}: shape {ssh_da.shape[-2:]} doesn't match "
                f"domain_cfg's T-grid {self._lat.shape}"
            )

        # mask land at U/V-points before colocation, so a land neighbor's non-NaN value can't
        # get averaged into an adjacent ocean T-point
        u_da = u_da.where(self._u_mask)
        v_da = v_da.where(self._v_mask)

        u_t, v_t = _to_t_point(u_da, v_da)
        u_east, v_north = unrotate_to_geographic(u_t, v_t, self._gcost, self._gsint)

        merged = xr.Dataset(
            {self.ssh_var: ssh_da, self.u_var: u_east, self.v_var: v_north},
            coords={
                "lat": (("y", "x"), self._lat),
                "lon": (("y", "x"), self._lon),
            },
        )
        merged = merged.rename({"time_counter": "time"})

        # NaN land T-point
        merged = merged.where(self._ocean_mask)

        return [merged]
