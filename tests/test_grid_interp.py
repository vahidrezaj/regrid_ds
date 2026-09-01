'''Tests for RegridPipeline, _rotate_vectors, and RegridPipeline._build_var_groups'''

import numpy as np
import pytest
import xarray as xr

from hbm_prep.grid_interp import (
    RegridPipeline,
    _rotate_vectors,
    create_local_metric_grid,
)

LAT_0, LON_0 = 60.0, 20.0
# regular lat/lon ranges (start, stop, step) comfortably covering the target
# grids built by _target_grid() below, so bilinear regridding never has to
# extrapolate outside the source's convex hull
SOURCE_LAT = (55, 66, 2)
SOURCE_LON = (5, 36, 2)


def _target_grid(domain_size_km=600, grid_size=7):
    return create_local_metric_grid(
        domain_size_km=domain_size_km, grid_size=grid_size,
        lat_0=LAT_0, lon_0=LON_0, proj_type="aeqd",
    )


def _source_ds(variable_names, values_by_time, lat_range=SOURCE_LAT, lon_range=SOURCE_LON):
    '''
    Small regular lat/lon source dataset (2-D "lat"/"lon" coords, dims "j","i"):
    each variable is spatially constant per timestep (one value per entry in
    `values_by_time`), so a correct bilinear regrid must return that same
    constant back -- a cheap, exact correctness check with no hand-derived
    interpolation math needed.
    '''
    lats = np.arange(*lat_range)
    lons = np.arange(*lon_range)
    lon2d, lat2d = np.meshgrid(lons, lats)

    data_vars = {
        var: (
            ("time", "j", "i"),
            np.stack([np.full(lat2d.shape, v, dtype=np.float64) for v in values_by_time]),
        )
        for var in variable_names
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": np.arange(len(values_by_time)),
            "lat": (("j", "i"), lat2d),
            "lon": (("j", "i"), lon2d),
        },
    )


def _build_pipeline(
    variable_names, interp_method="bilinear", extrap_method=None,
    pair_vars_list=None, use_mask=True, target_grid=None,
):
    return RegridPipeline(
        target_grid=target_grid if target_grid is not None else _target_grid(),
        variable_names=variable_names,
        interp_method=interp_method,
        extrap_method=extrap_method,
        pair_vars_list=pair_vars_list or [],
        use_mask=use_mask,
    )


# ---- RegridPipeline._build_var_groups ----------------------------------

def test_build_var_groups_single_group_for_shared_method():
    groups = RegridPipeline._build_var_groups(["sst", "ssh"], "bilinear", None)
    assert groups == [(["sst", "ssh"], "bilinear", None)]


def test_build_var_groups_one_group_per_variable_for_interp_list():
    groups = RegridPipeline._build_var_groups(
        ["sst", "ssh"], ["bilinear", "nearest_s2d"], None
    )
    assert groups == [(["sst"], "bilinear", None), (["ssh"], "nearest_s2d", None)]


def test_build_var_groups_extrap_list_forces_per_variable_split():
    groups = RegridPipeline._build_var_groups(
        ["sst", "ssh"], "bilinear", ["nearest_s2d", None]
    )
    assert groups == [(["sst"], "bilinear", "nearest_s2d"), (["ssh"], "bilinear", None)]


def test_build_var_groups_rejects_interp_length_mismatch():
    with pytest.raises(ValueError):
        RegridPipeline._build_var_groups(["sst", "ssh"], ["bilinear"], None)


def test_build_var_groups_rejects_extrap_length_mismatch():
    with pytest.raises(ValueError):
        RegridPipeline._build_var_groups(["sst", "ssh"], "bilinear", ["nearest_s2d"])


# ---- _rotate_vectors -----------------------------------------------------

def test_rotate_vectors_matches_cos_sin_and_keeps_attrs():
    target_grid = _target_grid()
    ds = xr.Dataset({
        "u": (("y", "x"), np.ones(target_grid["lat"].shape), {"units": "m/s"}),
        "v": (("y", "x"), np.zeros(target_grid["lat"].shape), {"units": "m/s"}),
    })

    rotated = _rotate_vectors(ds, ("u", "v"), target_grid)

    assert np.allclose(rotated["u"].values, target_grid["cos_g"].values)
    assert np.allclose(rotated["v"].values, target_grid["sin_g"].values)
    assert rotated["u"].attrs == {"units": "m/s"}
    assert rotated["v"].attrs == {"units": "m/s"}


# ---- RegridPipeline.__call__ ---------------------------------------------

def test_call_regrids_constant_field_to_same_constant():
    pipeline = _build_pipeline(["sst"])
    ds = _source_ds(["sst"], values_by_time=[10.0, 20.0])

    result = pipeline(ds_list=[ds], time_mask=np.array([True, False]))

    assert np.allclose(result["sst"].values, 10.0)


def test_call_static_time_mask_none():
    pipeline = _build_pipeline(["sst"])
    ds = _source_ds(["sst"], values_by_time=[42.0]).isel(time=0, drop=True)

    result = pipeline(ds_list=[ds], time_mask=None)

    assert np.allclose(result["sst"].values, 42.0)


def test_call_caches_regridder_and_still_reflects_new_data():
    pipeline = _build_pipeline(["sst"])
    ds = _source_ds(["sst"], values_by_time=[10.0, 20.0])

    result1 = pipeline(ds_list=[ds], time_mask=np.array([True, False]))
    assert len(pipeline._regridder_cache) == 1
    cached_regridder = pipeline._regridder_cache[(0, 0)]

    result2 = pipeline(ds_list=[ds], time_mask=np.array([False, True]))
    assert len(pipeline._regridder_cache) == 1
    assert pipeline._regridder_cache[(0, 0)] is cached_regridder

    assert np.allclose(result1["sst"].values, 10.0)
    assert np.allclose(result2["sst"].values, 20.0)


def test_call_per_variable_interp_method_caches_one_regridder_per_group():
    pipeline = _build_pipeline(["sst", "ssh"], interp_method=["bilinear", "nearest_s2d"])
    ds = _source_ds(["sst", "ssh"], values_by_time=[5.0])

    result = pipeline(ds_list=[ds], time_mask=np.array([True]))

    assert set(pipeline._regridder_cache) == {(0, 0), (0, 1)}
    assert np.allclose(result["sst"].values, 5.0)
    assert np.allclose(result["ssh"].values, 5.0)


def test_call_caches_region_mask_and_applies_it_on_every_call():
    target_grid = _target_grid()
    pipeline = _build_pipeline(["sst"], target_grid=target_grid)

    lats = np.arange(*SOURCE_LAT)
    lons = np.arange(*SOURCE_LON)
    lon2d, lat2d = np.meshgrid(lons, lats)
    land = lat2d < LAT_0  # "land" mask, stable across calls (see class caching assumption)

    def make_ds(value):
        arr = np.where(land, np.nan, value)[None, ...]
        return xr.Dataset(
            {"sst": (("time", "j", "i"), arr)},
            coords={"time": [0], "lat": (("j", "i"), lat2d), "lon": (("j", "i"), lon2d)},
        )

    result1 = pipeline(ds_list=[make_ds(10.0)], time_mask=np.array([True]))
    assert len(pipeline._masks_cache) == 1
    cached_masks = pipeline._masks_cache[0]

    result2 = pipeline(ds_list=[make_ds(20.0)], time_mask=np.array([True]))
    assert len(pipeline._masks_cache) == 1
    assert pipeline._masks_cache[0] is cached_masks

    for result, value in ((result1, 10.0), (result2, 20.0)):
        values = result["sst"].values
        assert np.any(np.isnan(values))  # masked-out "land" cells
        assert np.any(np.isclose(values[~np.isnan(values)], value))


def test_call_mosaics_regions_by_priority():
    target_grid = _target_grid()
    pipeline = _build_pipeline(["sst"], target_grid=target_grid)

    lats = np.arange(*SOURCE_LAT)
    lons = np.arange(*SOURCE_LON)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # higher-priority region only reports data north of the domain center
    # (NaN south of it, like a regional product with partial coverage)
    region_a = xr.Dataset(
        {"sst": (("time", "j", "i"), np.where(lat2d < LAT_0, np.nan, 100.0)[None, ...])},
        coords={"time": [0], "lat": (("j", "i"), lat2d), "lon": (("j", "i"), lon2d)},
    )
    # lower-priority region covers the whole domain
    region_b = _source_ds(["sst"], [200.0], lat_range=SOURCE_LAT, lon_range=SOURCE_LON)

    result = pipeline(ds_list=[region_a, region_b], time_mask=np.array([True]))
    values = result["sst"].values

    assert not np.isnan(values).any()          # region_b fills every gap left by region_a's mask
    assert np.any(np.isclose(values, 100.0))   # region_a (higher priority) wins somewhere
    assert np.any(np.isclose(values, 200.0))   # region_b fills the rest


def test_call_rotates_pair_vars_after_regridding():
    target_grid = _target_grid()
    pipeline = _build_pipeline(["u", "v"], pair_vars_list=[("u", "v")], target_grid=target_grid)

    ds = _source_ds(["u", "v"], values_by_time=[1.0])
    ds["v"] = xr.zeros_like(ds["v"])  # pure-u field: (u, v) = (1, 0) everywhere

    result = pipeline(ds_list=[ds], time_mask=np.array([True]))

    assert np.allclose(result["u"].values, target_grid["cos_g"].values)
    assert np.allclose(result["v"].values, target_grid["sin_g"].values)
