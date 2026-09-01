'''
functions for regriding dataset
'''
from functools import reduce

import numpy as np
import xarray as xr
import xesmf as xe
from pyproj import CRS, Transformer, Proj


def create_local_metric_grid(
        domain_size_km: float,
        grid_size: int,
        lat_0: float,
        lon_0: float,
        proj_type: str = "aeqd",
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
    
    Return:
    ----------
    lat, lon, y, x, cos_g, sin_g, lon_grid_b, lat_grid_b

    Limitations
    ----------
    `cos_g`/`sin_g` are exact only at (lat_0, lon_0); AEQD/LAEA aren't conformal off-center,
    so vector rotation is rotation-only (no shear/scale correction). Verified negligible up
    to domain_size_km ~1500 (angular error <~0.3 deg); re-check for larger domains.

    """
    half_domain_m = (domain_size_km * 1000.0) / 2.0

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

    # Transform grid to lat/lon & extract factors (includes convergence angle gamma)
    lon_grid, lat_grid = inv.transform(x_mg, y_mg)
    lon_grid_b, lat_grid_b = inv.transform(xx_b, yy_b)

    # compute convergence angles:
    p = Proj(f"+proj={proj_type} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m")
    factors = p.get_factors(lon_grid, lat_grid)
    gamma_rad = np.deg2rad(factors.meridian_convergence)

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
    }
    return out


def _create_masks(use_mask, sample_array, target_grid, thrd_ocean_fraction=0.5):
    '''
    Create source and target mask from `var_name` variable.
    If the variable doesn't contain NaN values, the function returns None

    Returns: source_mask, target_mask
    '''
    if use_mask and sample_array.isnull().any():
        ds_source = sample_array.to_dataset(name="var")

        ds_target = xr.Dataset(
            coords={
                "lat": (("y", "x"), target_grid['lat']),
                "lon": (("y", "x"), target_grid['lon']),
            }
        )
        regridder = xe.Regridder(
            ds_source,
            ds_target,
            "bilinear",
            unmapped_to_nan=True,
        )

        source_mask = (~sample_array.isnull()).astype(float)

        target_ocean_fraction = regridder(source_mask)
        target_mask = target_ocean_fraction > thrd_ocean_fraction

        return source_mask.astype(int), target_mask.astype(int)

    else:
        return None, None

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
    Regrid and mosaic one or more source datasets onto a target grid, then rotate
    vector variables into the target grid's local basis -- call repeatedly with
    `ds_list`/`time_mask` (e.g. one call per file in `HBMPreProcessing`'s
    read/regrid/write loop) to get back a mosaiced, regridded, vector-rotated
    `xr.Dataset` each time. Caches per-region xesmf regridders and land/ocean
    masks across those calls instead of rebuilding them every time.

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
    extrap_method : list or str or None
        Extrapolation method, one per variable if a list. `None` (the default)
        keeps land/edges as NaN; `nearest_s2d`/`inverse_dist` fill beyond the
        source domain (e.g. for atmospheric data). Forced to `None` for any
        region whose source mask is not `None` (masking and extrapolation are
        mutually exclusive strategies for handling land/edges).
    pair_vars_list : list of (str, str)
        (u, v) variable name pairs, already regridded, to rotate from true
        north/east into the target grid's local basis via `_rotate_vectors`.
    use_mask : bool, default True
        Whether NaNs in the source are a real land/ocean mask to respect (excluded
        as regridding sources, and forcing `extrap_method` to `None`). Set to
        `False` for sources whose NaNs are just incomplete domain coverage, so
        that `extrap_method` fills the gap instead of being silently disabled.

    Notes
    -----
    Caching assumes each region's source lat/lon coordinates and land/ocean mask
    are stable across calls -- only the underlying data values are expected to
    change between calls.
    '''

    def __init__(
        self,
        target_grid,
        variable_names,
        interp_method,
        extrap_method,
        pair_vars_list,
        use_mask=True,
    ):
        self.target_grid = target_grid
        self.variable_names = list(variable_names)
        self.pair_vars_list = pair_vars_list
        self.use_mask = use_mask

        # one (group_variable_names, interp_method, extrap_method) tuple per
        # xe.Regridder to build for each region, computed (and validated) once
        # here instead of on every call
        self._var_groups = self._build_var_groups(
            self.variable_names, interp_method, extrap_method
        )

        # region_idx -> (source_mask, target_mask)
        self._masks_cache = {}
        # (region_idx, group_idx) -> xe.Regridder, group_idx indexing self._var_groups
        self._regridder_cache = {}

    @staticmethod
    def _build_var_groups(variable_names, interp_method, extrap_method):
        '''
        Precompute the (group_variable_names, interp_method, extrap_method) groups
        to build one `xe.Regridder` per: a single group covering all variables
        when `interp_method` is a plain string and `extrap_method` isn't a list,
        otherwise one single-variable group per entry in `variable_names`. Raises
        `ValueError` on an interp_method/extrap_method list whose length doesn't
        match `variable_names`.
        '''
        if isinstance(interp_method, str) and not isinstance(extrap_method, list):
            return [(list(variable_names), interp_method, extrap_method)]

        if isinstance(interp_method, list):
            if len(interp_method) != len(variable_names):
                raise ValueError(
                    f"interp_method ({len(interp_method)}) must have the same length "
                    f"as variable_names ({len(variable_names)})"
                )
        else:
            interp_method = [interp_method] * len(variable_names)

        if isinstance(extrap_method, list):
            if len(extrap_method) != len(variable_names):
                raise ValueError(
                    f"extrap_method ({len(extrap_method)}) must have the same length "
                    f"as variable_names ({len(variable_names)})"
                )
        else:
            extrap_method = [extrap_method] * len(variable_names)

        return [
            ([var], method, extrap)
            for var, method, extrap in zip(variable_names, interp_method, extrap_method)
        ]

    def _region_masks(self, ds, region_idx):
        '''
        Return this region's (source_mask, target_mask) pair (see `_create_masks`),
        building and caching it by `region_idx` the first time this region is
        seen. Caching assumes each region's land/ocean mask is stable across
        calls -- see class docstring.
        '''
        if region_idx not in self._masks_cache:
            sample_array = ds[self.variable_names[0]]
            sample_array = (
                sample_array.isel(time=0) if "time" in sample_array.dims else sample_array
            )
            self._masks_cache[region_idx] = _create_masks(
                self.use_mask, sample_array, self.target_grid, thrd_ocean_fraction=0.5
            )
        return self._masks_cache[region_idx]

    def _build_regridder(self, ds_source, interp_method, masks, extrap_method):
        '''
        Build one `xe.Regridder` from `ds_source` (already restricted to one
        group's variables plus `lat`/`lon`) onto `self.target_grid`. This is the
        (expensive, weight-computing) step `_regrid_region` caches so it only
        runs once per `(region_idx, group_idx)` instead of on every call.
        '''
        if masks[0] is not None:
            ds_source["mask"] = (("lat", "lon"), masks[0].values)

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
        if masks[1] is not None:
            ds_target["mask"] = (("lat", "lon"), masks[1].values)

        return xe.Regridder(
            ds_source,
            ds_target,
            interp_method,
            extrap_method=extrap_method,
            ignore_degenerate=True,
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

        masks = self._region_masks(ds, region_idx)

        ds_regridded = []
        for group_idx, (group_vars, interp_method, extrap_method) in enumerate(self._var_groups):
            ds_source = xr.Dataset(
                {var: ds[var] for var in group_vars},
                coords={"lat": ds.lat, "lon": ds.lon},
            )

            cache_key = (region_idx, group_idx)
            regridder = self._regridder_cache.get(cache_key)
            if regridder is None:
                # masking and extrapolation are mutually exclusive (see class docstring)
                build_extrap = None if masks[0] is not None else extrap_method
                regridder = self._build_regridder(ds_source, interp_method, masks, build_extrap)
                self._regridder_cache[cache_key] = regridder

            ds_group = regridder(ds_source[group_vars], keep_attrs=True)
            if masks[1] is not None:
                for var in group_vars:
                    ds_group[var] = ds_group[var].where(masks[1] > 0.5)
            ds_regridded.append(ds_group)

        return xr.merge(ds_regridded) if len(ds_regridded) > 1 else ds_regridded[0]

    def __call__(self, ds_list, time_mask):
        '''
        Regrid and mosaic `ds_list` (one dataset per region, in PRIORITY order:
        where sources overlap on the target grid, `ds_list[0]`'s data wins, later
        datasets only fill gaps left by earlier ones) onto `self.target_grid`,
        then rotate `self.pair_vars_list` vector variables into the target grid's
        local basis.

        Parameters
        ----------
        ds_list : list of xr.Dataset
            Source datasets, in PRIORITY order (see above).
        time_mask : array-like of bool, or None
            Boolean mask selecting time steps to keep, applied before regridding.
            `None` means the sources have no time axis at all (e.g. a static
            bathymetry raster): each source is regridded as-is, with no
            time/depth trimming. Otherwise, only the first depth level (if a
            variable is 3-D) and the time steps selected by `time_mask` are kept
            before regridding.

        Returns
        -------
        xr.Dataset
            Mosaiced, regridded dataset on the target grid with vectors rotated.
        '''
        ds_regridded = []
        for region_idx, ds in enumerate(ds_list):
            if time_mask is not None:
                # select the first depth index, if var is 3D
                ds = ds.map(lambda da: da[time_mask, 0] if da.ndim > 3 else da[time_mask])
            ds_regridded.append(self._regrid_region(ds, region_idx))

        # mosaic regridded regions by priority, if len(ds_regridded)>1
        if len(ds_regridded) > 1:
            ds = reduce(
                lambda base, nxt: base.combine_first(nxt), ds_regridded[1:], ds_regridded[0]
            )
        else:
            ds = ds_regridded[0]

        # rotate vector variables:
        for pair_vars in self.pair_vars_list:
            ds = _rotate_vectors(ds, pair_vars, self.target_grid)

        return ds
