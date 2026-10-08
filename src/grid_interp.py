'''
Target grid and regridding: 
    regrid source regions, mosaic them, mask land, fill gaps and rotate vectors.
'''
from functools import reduce

import numpy as np
import xarray as xr
import xesmf as xe
from pyproj import CRS, Transformer, Proj

from spatial_fill import GapFiller


def create_local_metric_grid(
        domain_size_km: float,
        grid_size: int,
        lat_0: float,
        lon_0: float,
        proj_type: str = "aeqd",
        alpha_deg: float = 0.0,
) -> dict:
    """
    Generate lat/lon coordinates corresponding to an equidistant, uniform Cartesian grid
    centered dynamically at (lat_0, lon_0).

    Parameters
    ----------
    domain_size_km : float
        Side length of the square domain in kilometers (e.g., 1000).
    grid_size : int
        Number of grid cells along each dimension.
    lat_0, lon_0 : float
        Center latitude and longitude of the moving window.
    proj_type : str
        - 'aeqd' (Azimuthal Equidistant - best for distance/FFT)
        - 'laea' (Equal Area - conserving spatial integral properties of scalar fields).
    alpha_deg : float, default 0.0
        Tilt the grid's +y ("up") axis this many degrees clockwise from true north,
        instead of aligning it to true north. AEQD/LAEA are symmetric around their
        center, so a rotated grid is just as valid as an unrotated one. Useful for
        boxing a tilted source domain more tightly, with less wasted (NaN) area.
        `alpha_deg=0.0` (default) is the original, unrotated grid.

    Return:
    ----------
    lat, lon, y, x, cos_g, sin_g, lon_grid_b, lat_grid_b, crs, alpha_deg

    Limitations
    ----------
    `cos_g`/`sin_g` are exact only at (lat_0, lon_0); AEQD/LAEA aren't conformal off-center,
    so vector rotation is rotation-only (no shear/scale correction). Verified negligible up
    to domain_size_km ~1500 (angular error <~0.3 deg); re-check for larger domains. `alpha_deg`
    doesn't change this. It's a constant angular offset, not a new source of off-center error.

    """
    half_domain_m = (domain_size_km * 1000.0) / 2.0
    alpha_rad = np.deg2rad(alpha_deg)
    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)

    # Define dynamic projection centered on window
    proj_crs = CRS.from_proj4(
        f'+proj={proj_type} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m'
    )
    geo_crs = CRS.from_epsg(4326)

    # Coordinate transformer
    inv = Transformer.from_crs(proj_crs, geo_crs, always_xy=True)

    # Uniform metric Cartesian coordinates (Metres)
    axis = np.linspace(-half_domain_m, half_domain_m, grid_size)

    dx = axis[1] - axis[0]
    # Cell edges in projected coordinates
    axis_b = np.concatenate([
        [axis[0] - dx / 2],
        (axis[:-1] + axis[1:]) / 2,
        [axis[-1] + dx / 2],
    ])

    # 2-D center coordinates
    x_mg, y_mg = np.meshgrid(axis, axis)

    # 2-D corner coordinates
    xx_b, yy_b = np.meshgrid(axis_b, axis_b)

    # Rotate nominal grid coordinates into the AEQD projection's own native
    # (unrotated) frame before inverse-transforming; identity when alpha_deg=0
    x_native = x_mg * cos_a + y_mg * sin_a
    y_native = -x_mg * sin_a + y_mg * cos_a
    xb_native = xx_b * cos_a + yy_b * sin_a
    yb_native = -xx_b * sin_a + yy_b * cos_a

    # Transform grid to lat/lon & extract factors (includes convergence angle gamma)
    lon_grid, lat_grid = inv.transform(x_native, y_native)
    lon_grid_b, lat_grid_b = inv.transform(xb_native, yb_native)

    # compute convergence angles:
    p = Proj(f"+proj={proj_type} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m")
    factors = p.get_factors(lon_grid, lat_grid)
    # the grid's own rotation adds directly to the (position-dependent) meridian
    # convergence: alpha is a single constant offset applied uniformly on top of it
    gamma_rad = np.deg2rad(factors.meridian_convergence) + alpha_rad

    # Rotate vectors using standard matrix
    cos_g = np.cos(gamma_rad)
    sin_g = np.sin(gamma_rad)

    out = {
        'lat': lat_grid,
        'lon': lon_grid,
        'y': axis,
        'x': axis,
        'cos_g': xr.DataArray(cos_g, dims=("y", "x")),
        'sin_g': xr.DataArray(sin_g, dims=("y", "x")),
        'lat_b': lat_grid_b,
        'lon_b': lon_grid_b,
        'crs': proj_crs.to_cf(),
        'alpha_deg': alpha_deg,
    }
    return out


def _rotate_vectors(ds, pair_vars, target_grid):
    '''
    Rotate vectors
    
    pair_vars: list of paired vector variables (u, v) aligned toward true east and north
    target_grid: contains cos_g and sin_g, `np.deg2rad(factors.meridian_convergence)`
    '''
    u, v = pair_vars
    u_attrs, v_attrs = ds[u].attrs, ds[v].attrs

    u_rot = ds[u] * target_grid['cos_g'] - ds[v] * target_grid['sin_g']
    v_rot = ds[u] * target_grid['sin_g'] + ds[v] * target_grid['cos_g']
    u_rot.attrs, v_rot.attrs = u_attrs, v_attrs

    ds[u], ds[v] = u_rot, v_rot

    return ds


class RegridPipeline:
    '''
    Regrid source regions onto the target grid, mosaic them, mask land with NaN,
    fill remaining gaps, and rotate vectors into the grid's local x/y axes. Built
    once, called per file.
    xesmf regridders are cached across calls.

    Parameters
    ----------
    target_grid : dict
        Target grid, as returned by `create_local_metric_grid`.
    variable_names : list
        Variables to regrid (e.g., ['sst', 'ssh', 'u', 'v']).
    interp_method : list or str
        Interpolation method, one per variable if a list; a plain string applies
        to all variables via a single shared `xe.Regridder`. See xESMF docs --
        one of 'bilinear', 'conservative', 'conservative_normed', 'patch',
        'nearest_s2d', 'nearest_d2s'.
    pair_vars_list : list of (str, str)
        (u, v) variable name pairs, already regridded, to rotate from true
        north/east into the target grid's local basis via `_rotate_vectors`.
    land_mask : (y, x) bool array or None, default None
        True = land: set to NaN, never filled (see `land_mask.py`). None = no land (forcing).
    fill_method : str or None, default None
        "nearest" or "laplace" (see `spatial_fill.py`). None leaves gaps as NaN.

    Attributes
    ----------
    valid_mask : (y, x) bool array or None
        True where every variable had real (not filled) data in every step so far.
        Only tracked with a land mask; None for forcing.

    '''

    def __init__(
        self,
        target_grid,
        variable_names,
        interp_method,
        pair_vars_list,
        land_mask=None,
        fill_method=None,
    ):
        self.target_grid = target_grid
        self.variable_names = list(variable_names)
        self.pair_vars_list = pair_vars_list
        self.land_mask = None if land_mask is None else np.asarray(land_mask, dtype=bool)
        self.gap_filler = GapFiller(self.land_mask, fill_method) if fill_method else None
        self.valid_mask = None

        # (variables, interp_method) per xe.Regridder, checked once here
        self._var_groups = self._build_var_groups(self.variable_names, interp_method)

        # (region_idx, group_idx) -> xe.Regridder, group_idx indexing self._var_groups
        self._regridder_cache = {}

    @staticmethod
    def _build_var_groups(variable_names, interp_method):
        ''' one group for all variables if interp_method is a string, else one per variable '''
        if isinstance(interp_method, str):
            return [(list(variable_names), interp_method)]

        if len(interp_method) != len(variable_names):
            raise ValueError(
                f"interp_method ({len(interp_method)}) must have the same length "
                f"as variable_names ({len(variable_names)})"
            )
        return [([var], method) for var, method in zip(variable_names, interp_method)]

    def _build_regridder(self, ds_source, interp_method):
        '''
        Build one `xe.Regridder` from `ds_source` (already restricted to one
        group's variables plus `lat`/`lon`) onto `self.target_grid`. This is the
        (expensive, weight-computing) step `_regrid_region` caches so it only
        runs once per `(region_idx, group_idx)` instead of on every call.
        '''
        if interp_method in ("conservative", "conservative_normed"):
            target_vars = {
                "lat_b": (("y_b", "x_b"), self.target_grid['lat_b']),
                "lon_b": (("y_b", "x_b"), self.target_grid['lon_b']),
            }
        else:
            target_vars = {}

        ds_target = xr.Dataset(
            target_vars,
            coords={
                "lat": (("y", "x"), self.target_grid['lat']),
                "lon": (("y", "x"), self.target_grid['lon']),
            },
        )
        # unmapped_to_nan must be True: the default (0 outside the source) breaks the mosaic
        return xe.Regridder(
            ds_source, ds_target, interp_method, ignore_degenerate=True, unmapped_to_nan=True,
        )

    def _regrid_region(self, ds, region_idx):
        '''
        Regrid one region's dataset onto `self.target_grid`, group by group (see
        `_build_var_groups`), building (and caching, by `(region_idx, group_idx)`)
        each group's `xe.Regridder` the first time it's needed and just applying
        it on every later call.
        '''
        missing = set(self.variable_names) - set(ds.data_vars)
        if missing:
            raise ValueError(f"Variables {missing} not in dataset")

        ds_regridded = []
        for group_idx, (group_vars, interp_method) in enumerate(self._var_groups):
            ds_source = xr.Dataset(
                {var: ds[var] for var in group_vars},
                coords={"lat": ds.lat, "lon": ds.lon},
            )
            cache_key = (region_idx, group_idx)
            regridder = self._regridder_cache.get(cache_key)
            if regridder is None:
                regridder = self._build_regridder(ds_source, interp_method)
                self._regridder_cache[cache_key] = regridder
            ds_regridded.append(regridder(ds_source[group_vars], keep_attrs=True))

        return xr.merge(ds_regridded) if len(ds_regridded) > 1 else ds_regridded[0]

    def _mask_and_fill(self, ds):
        ''' NaN out land, update valid_mask, then fill the other NaN cells '''
        for var in self.variable_names:
            values = ds[var].values
            if self.land_mask is not None:
                values = np.where(self.land_mask, np.nan, values)
                valid = np.isfinite(values).reshape(-1, *self.land_mask.shape).all(axis=0)
                self.valid_mask = valid if self.valid_mask is None else self.valid_mask & valid
            if self.gap_filler is not None:
                values = self.gap_filler(values)
            ds[var] = ds[var].copy(data=values)
        return ds

    def __call__(self, ds_list, time_mask):
        '''
        Regrid and mosaic `ds_list` (one dataset per region, in PRIORITY order:
        where sources overlap on the target grid, `ds_list[0]`'s data wins, later
        datasets only fill gaps left by earlier ones) onto `self.target_grid`,
        mask land, fill gaps, then rotate `self.pair_vars_list` into the grid's x/y.

        Parameters
        ----------
        ds_list : list of xr.Dataset
            Source datasets, in PRIORITY order (see above).
        time_mask : array-like of bool, or None
            Boolean mask selecting time steps to keep, applied before regridding.
            `None` means the sources have no time axis at all (e.g. a static
            bathymetry raster): each source is regridded as-is, with no
            time trimming. Otherwise, only the time steps selected by `time_mask`
            are kept before regridding. Variables must already be (time, y, x) --
            level dims are dropped by the reader (see `readers.select_first_level`).

        Returns
        -------
        xr.Dataset
            On the target grid: mosaiced, land masked, gaps filled, vectors rotated.
        '''
        ds_regridded = []
        for region_idx, ds in enumerate(ds_list):
            if time_mask is not None:
                ds = ds.isel(time=np.asarray(time_mask))
            ds_regridded.append(self._regrid_region(ds, region_idx))

        # mosaic regridded regions by priority, if len(ds_regridded)>1
        if len(ds_regridded) > 1:
            ds = reduce(
                lambda base, nxt: base.combine_first(nxt), ds_regridded[1:], ds_regridded[0]
            )
        else:
            ds = ds_regridded[0]

        # fill before rotating, so u/v are filled as east/north
        ds = self._mask_and_fill(ds)

        # rotate vector variables:
        for pair_vars in self.pair_vars_list:
            ds = _rotate_vectors(ds, pair_vars, self.target_grid)

        return ds
