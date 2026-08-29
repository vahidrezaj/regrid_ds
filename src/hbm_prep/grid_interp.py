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

def _regrid_helper(ds, target_grid, selected_vars:list, interp_method, masks, extrap_method):
    '''
    regridding helper function
    '''
    # source dataset:
    ds_source = xr.Dataset(
        {var: ds[var] for var in selected_vars},
        coords={"lat": ds.lat, "lon": ds.lon}
    )
    if masks[0] is not None:
        ds_source["mask"] = (("lat", "lon"), masks[0].values)

    # target dataset:
    if interp_method in ("conservative", "conservative_normed"):
        target_vars = {
            "lat_b": (("y_b", "x_b"), target_grid['lat_b']),
            "lon_b": (("y_b", "x_b"), target_grid['lon_b']),
        }
    else:
        target_vars = {}

    ds_target = xr.Dataset(
        target_vars,
        coords={
            "lat": (("y", "x"), target_grid['lat']),
            "lon": (("y", "x"), target_grid['lon']),
        },
    )
    if masks[1] is not None:
        ds_target["mask"] = (("lat", "lon"), masks[1].values)

    # regrider:
    regridder = xe.Regridder(
        ds_source,
        ds_target,
        interp_method,
        extrap_method= extrap_method,
        ignore_degenerate= True,
    )

    ds_regridded = regridder(ds_source[selected_vars], keep_attrs=True)

    if masks[1] is not None:
        for var in selected_vars:
            ds_regridded[var] = ds_regridded[var].where(masks[1] > 0.5)

    return ds_regridded


def regrid_xesmf(
    ds: xr.Dataset,
    target_grid,
    variable_names: list,
    interp_method: list|str,
    masks: list,
    extrap_method: list|str|None= None,
) -> xr.Dataset:
    """
    Regrid multiple variables in a Dataset.

    If `interp_method`/`extrap_method` are both a single value (not a list), all
    variables share one `xe.Regridder` call; if either is a list, variables are
    regridded one at a time (own regridder each) and merged.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with variables to regrid
    target_grid : dict
        Target grid dict contains `lat`, `lon`, `lat_b`, and `lon_b`
    variable_names : list
        Variables to regrid (e.g., ['sst', 'ssh'])
    interp_method : list or str
        interpolation method, one per variable if a list. See xESMF documentation.
        interp_method can be: 'bilinear', 'conservative', 'conservative_normed', 'patch',
        'nearest_s2d', and 'nearest_d2s'
    masks : list
        [source_mask, target_mask] pair (each `None` or an `xr.DataArray`), as
        returned by `_create_masks`. If `masks[0]` is not None, `extrap_method`
        is forced to `None` regardless of the argument passed in, since masking
        and extrapolation are mutually exclusive strategies for handling land/edges.
    extrap_method : str, list or None, default is None
        Ignored (forced to None) whenever `masks[0]` is not None.
        None (keep land/edges as NaN) does not apply extrapolation.
        `nearest_s2d` or `inverse_dist` (fill beyond domain for atmospheric data)

    Returns
    -------
    xr.Dataset
        Regridded dataset with `variable_names` on the target grid.
    """

    # Validate variables exist
    missing = set(variable_names) - set(ds.data_vars)
    if missing:
        raise ValueError(f"Variables {missing} not in dataset")

    # force extrap_method to None if we have masks
    if masks[0] is not None:
        extrap_method = None

    if isinstance(interp_method, str) and not isinstance(extrap_method, list):
        # one regrider for all variables:
        return _regrid_helper(ds, target_grid, variable_names, interp_method, masks, extrap_method)

    else:
        # loop variable_names:
        if isinstance(interp_method, list):
            assert len(interp_method) == len(variable_names), (f"interp_method "
                f"({len(interp_method)}) must have the same length as "
                f"variable_names ({len(variable_names)})")
        else:
            interp_method = [interp_method,] * len(variable_names)

        if isinstance(extrap_method, list):
            assert len(extrap_method) == len(variable_names), (f"extrap_method "
                f"({len(extrap_method)}) must have the same length as "
                f"variable_names ({len(variable_names)})")
        else:
            extrap_method = [extrap_method,] * len(variable_names)

        # regridding loop:
        ds_regridded = []
        for var, method, extrap in zip(variable_names, interp_method, extrap_method):
            ds_regridded.append(_regrid_helper(ds, target_grid, [var], method, masks, extrap))

        return xr.merge(ds_regridded)


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


def regridding_fn(
        ds_list,
        target_grid,
        variable_names,
        time_mask,
        interp_method,
        extrap_method,
        pair_vars_list,
        use_mask=True,
) -> xr.Dataset:
    '''
    Regrid and mosaic one or more source datasets onto a target grid, then rotate
    vector variables into the target grid's local basis.

    Parameters
    ----------
    ds_list : list of xr.Dataset
        Source datasets, in PRIORITY order: where sources overlap on the target
        grid, `ds_list[0]`'s data wins, later datasets only fill gaps left by
        earlier ones.
    target_grid : dict
        Target grid, as returned by `create_local_metric_grid`.
    variable_names : list
        Variables to regrid (e.g., ['sst', 'ssh', 'u', 'v']).
    time_mask : array-like of bool, or None
        Boolean mask selecting time steps to keep, applied before regridding.
        `None` means the sources have no time axis at all (e.g. a static
        bathymetry raster): each source is regridded as-is, with no
        time/depth trimming.
    interp_method : list or str
        Passed through to `regrid_xesmf` (e.g. 'bilinear', 'conservative').
    extrap_method : list or str or None
        Passed through to `regrid_xesmf`; ignored per-source when that source's
        mask is not None (see `regrid_xesmf`).
    pair_vars_list : list of (str, str)
        (u, v) variable name pairs, already regridded, to rotate from true
        north/east into the target grid's local basis via `_rotate_vectors`.
    use_mask : bool, default True
        Whether NaNs in the source are a real land/ocean mask to respect (excluded
        as regridding sources, and forcing `extrap_method` to `None`. Set to `False`
        for sources whose NaNs are just incomplete domain coverage, so that `extrap_method`
        fills the gap instead of being silently disabled.

    Returns
    -------
    xr.Dataset
        Mosaiced, regridded dataset on the target grid with vectors rotated.

    Notes
    -----
    For each source, only the first depth level and the time steps selected by
    `time_mask` are kept before regridding.
    '''
    ds_regridded = []
    for ds in ds_list:
        if time_mask is not None:
            # select the first depth index, if var is 3D
            ds = ds.map(lambda da: da[time_mask, 0] if da.ndim > 3 else da[time_mask])

        # create masks:
        sample_array = ds[variable_names[0]]
        sample_array = sample_array.isel(time=0) if "time" in sample_array.dims else sample_array
        masks = _create_masks(use_mask, sample_array, target_grid, thrd_ocean_fraction=0.5)

        # regriding:
        ds_regridded.append(
            regrid_xesmf(ds, target_grid, variable_names, interp_method, masks, extrap_method)
        )


    # mosaic regridded sources by priority, if len(ds_regridded)>1
    if len(ds_regridded) > 1:
        ds = reduce(lambda base, nxt: base.combine_first(nxt), ds_regridded[1:], ds_regridded[0])
    else:
        ds = ds_regridded[0]

    # rotate vector variables:
    for pair_vars in pair_vars_list:
        ds = _rotate_vectors(ds, pair_vars, target_grid)

    return ds
