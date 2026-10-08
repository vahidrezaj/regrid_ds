'''Tests for RegridPipeline, _rotate_vectors, RegridPipeline._build_var_groups,
and create_local_metric_grid's alpha_deg rotation'''

import numpy as np
import pytest
import xarray as xr

from grid_interp import (
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
    variable_names, interp_method="bilinear", pair_vars_list=None, target_grid=None,
    land_mask=None, fill_method=None,
):
    return RegridPipeline(
        target_grid=target_grid if target_grid is not None else _target_grid(),
        variable_names=variable_names,
        interp_method=interp_method,
        pair_vars_list=pair_vars_list or [],
        land_mask=land_mask,
        fill_method=fill_method,
    )


# ---- create_local_metric_grid / alpha_deg ---------------------------------

def test_alpha_deg_zero_matches_omitted_default():
    with_zero = create_local_metric_grid(
        domain_size_km=600, grid_size=7, lat_0=LAT_0, lon_0=LON_0, alpha_deg=0.0,
    )
    omitted = _target_grid()

    assert np.array_equal(with_zero["lat"], omitted["lat"])
    assert np.array_equal(with_zero["lon"], omitted["lon"])
    assert np.allclose(with_zero["cos_g"].values, omitted["cos_g"].values)
    assert np.allclose(with_zero["sin_g"].values, omitted["sin_g"].values)


def test_alpha_deg_rotates_grid_coordinates_correctly():
    ''' a rotated grid's nominal (x, y) point should land at the same lon/lat as
    manually rotating the coordinates into the AEQD-native frame and calling the
    existing (unrotated) transform directly -- an independent check of the
    rotation direction/sign, not just a self-consistency check. '''
    from pyproj import CRS, Transformer  # pylint: disable=import-outside-toplevel

    alpha_deg = 37.0
    alpha = np.deg2rad(alpha_deg)
    domain_size_km, grid_size = 600, 7

    grid = create_local_metric_grid(
        domain_size_km=domain_size_km, grid_size=grid_size,
        lat_0=LAT_0, lon_0=LON_0, alpha_deg=alpha_deg,
    )

    # a non-center, non-edge nominal grid point
    iy, ix = 5, 2
    x, y = grid["x"][ix], grid["y"][iy]

    x_native = x * np.cos(alpha) + y * np.sin(alpha)
    y_native = -x * np.sin(alpha) + y * np.cos(alpha)

    proj_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={LAT_0} +lon_0={LON_0} +datum=WGS84 +units=m")
    inv = Transformer.from_crs(proj_crs, CRS.from_epsg(4326), always_xy=True)
    lon_expected, lat_expected = inv.transform(x_native, y_native)

    assert np.isclose(grid["lon"][iy, ix], lon_expected)
    assert np.isclose(grid["lat"][iy, ix], lat_expected)


def test_alpha_deg_is_carried_through_in_returned_grid():
    ''' writers.py stashes this on `spatial_ref` as a discoverable (non-standard)
    attr, so it must round-trip through the returned dict unchanged. '''
    grid = create_local_metric_grid(
        domain_size_km=600, grid_size=7, lat_0=LAT_0, lon_0=LON_0, alpha_deg=8.5,
    )
    assert grid["alpha_deg"] == 8.5
    assert _target_grid()["alpha_deg"] == 0.0


def test_alpha_deg_shifts_cos_sin_at_center_exactly():
    ''' at the exact grid center, meridian convergence is 0, so cos_g/sin_g there
    should equal cos(alpha)/sin(alpha) exactly -- a tight, unambiguous sign check. '''
    alpha_deg = 15.0
    grid_size = 7  # odd -> an exact center point exists
    grid = create_local_metric_grid(
        domain_size_km=600, grid_size=grid_size, lat_0=LAT_0, lon_0=LON_0, alpha_deg=alpha_deg,
    )
    center = grid_size // 2

    assert np.isclose(
        grid["cos_g"].values[center, center], np.cos(np.deg2rad(alpha_deg)), atol=1e-6,
    )
    assert np.isclose(
        grid["sin_g"].values[center, center], np.sin(np.deg2rad(alpha_deg)), atol=1e-6,
    )


# ---- RegridPipeline._build_var_groups ----------------------------------

def test_build_var_groups_single_group_for_shared_method():
    groups = RegridPipeline._build_var_groups(["sst", "ssh"], "bilinear")
    assert groups == [(["sst", "ssh"], "bilinear")]


def test_build_var_groups_one_group_per_variable_for_interp_list():
    groups = RegridPipeline._build_var_groups(["sst", "ssh"], ["bilinear", "nearest_s2d"])
    assert groups == [(["sst"], "bilinear"), (["ssh"], "nearest_s2d")]


def test_build_var_groups_rejects_interp_length_mismatch():
    with pytest.raises(ValueError):
        RegridPipeline._build_var_groups(["sst", "ssh"], ["bilinear"])


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


def _gappy_ds(gaps, values):
    ''' one timestep per (gap mask, constant value) pair, NaN where the gap mask is True '''
    lats = np.arange(*SOURCE_LAT)
    lons = np.arange(*SOURCE_LON)
    lon2d, lat2d = np.meshgrid(lons, lats)
    arr = np.stack([np.where(g(lat2d, lon2d), np.nan, v) for g, v in zip(gaps, values)])
    return xr.Dataset(
        {"sst": (("time", "j", "i"), arr)},
        coords={
            "time": np.arange(len(values)),
            "lat": (("j", "i"), lat2d), "lon": (("j", "i"), lon2d),
        },
    )


def test_call_nans_land_and_fills_only_ocean_gaps():
    ''' land (from the shared mask) is NaN even where the source has data, and every
    other NaN -- here a coverage gap in the source -- is filled '''
    target_grid = _target_grid()
    land = target_grid["lon"] < LON_0 - 2
    pipeline = _build_pipeline(
        ["sst"], target_grid=target_grid, land_mask=land, fill_method="nearest",
    )
    ds = _gappy_ds([lambda lat, lon: (lat < LAT_0) & (lon > LON_0)], [10.0])

    values = pipeline(ds_list=[ds], time_mask=np.array([True]))["sst"].values[0]

    assert land.any() and np.isnan(values[land]).all()
    assert np.allclose(values[~land], 10.0)


def test_valid_mask_excludes_land_and_filled_gaps():
    ''' valid = real data before filling; a step with a bigger gap shrinks it '''
    target_grid = _target_grid()
    lat, lon = target_grid["lat"], target_grid["lon"]
    land = lon < LON_0 - 2
    pipeline = _build_pipeline(
        ["sst"], target_grid=target_grid, land_mask=land, fill_method="nearest",
    )
    south = lambda la, lo: (la < LAT_0) & (lo > LON_0)
    assert pipeline.valid_mask is None

    pipeline(ds_list=[_gappy_ds([south], [10.0])], time_mask=np.array([True]))
    first = pipeline.valid_mask.copy()
    assert not first[land].any()
    assert first[(lat > LAT_0 + 1) & ~land].all()
    assert not first[(lat < LAT_0 - 1) & (lon > LON_0 + 1)].any()  # filled, not valid

    # same pattern again: unchanged
    pipeline(ds_list=[_gappy_ds([south], [11.0])], time_mask=np.array([True]))
    np.testing.assert_array_equal(pipeline.valid_mask, first)

    # a step that also lacks the east: only cells valid in every step are kept
    east = lambda la, lo: lo > LON_0 + 3
    pipeline(ds_list=[_gappy_ds([south, east], [1.0, 2.0])], time_mask=np.ones(2, dtype=bool))
    assert (pipeline.valid_mask <= first).all() and pipeline.valid_mask.sum() < first.sum()


def test_valid_mask_not_tracked_without_land_mask():
    pipeline = _build_pipeline(["sst"], fill_method="nearest")
    pipeline(ds_list=[_source_ds(["sst"], [1.0])], time_mask=np.array([True]))
    assert pipeline.valid_mask is None


def test_call_without_fill_method_leaves_gaps_but_still_masks_land():
    target_grid = _target_grid()
    land = target_grid["lon"] < LON_0 - 2
    pipeline = _build_pipeline(["sst"], target_grid=target_grid, land_mask=land)
    ds = _gappy_ds([lambda lat, lon: (lat < LAT_0) & (lon > LON_0)], [10.0])

    values = pipeline(ds_list=[ds], time_mask=np.array([True]))["sst"].values[0]

    assert np.isnan(values[land]).all()
    assert np.isnan(values[~land]).any()  # the gap is still there


def test_call_fills_time_varying_gaps_without_land_mask():
    ''' no land mask (e.g. hbm_forcing): every NaN is a gap. The gap can move between
    timesteps (the north_sea hbm_forcing case), so each NaN pattern gets its own fill
    plan, and time order must be preserved. '''
    pipeline = _build_pipeline(["sst"], fill_method="nearest")
    south = lambda lat, lon: (lat < LAT_0) & (lon > LON_0)
    east = lambda lat, lon: lon > LON_0 + 3
    no_gap = lambda lat, lon: np.zeros_like(lat, dtype=bool)
    values = [1.0, 2.0, 3.0, 4.0]
    ds = _gappy_ds([south, east, no_gap, south], values)

    result = pipeline(ds_list=[ds], time_mask=np.ones(4, dtype=bool))

    assert list(result.time.values) == [0, 1, 2, 3]
    for t, value in enumerate(values):
        assert np.allclose(result["sst"].isel(time=t).values, value)  # no NaN left
    assert len(pipeline._regridder_cache) == 1  # one regridder, whatever the NaN pattern


@pytest.mark.parametrize("fill_method", [None, "nearest"])
def test_call_mosaics_regions_by_priority(fill_method):
    ''' with fill on too: filling runs after the mosaic, so region_a's gap is filled by
    region_b's data, not by extrapolating region_a over it '''
    target_grid = _target_grid()
    pipeline = _build_pipeline(["sst"], target_grid=target_grid, fill_method=fill_method)

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


def test_call_cells_outside_a_region_grid_are_nan_not_zero():
    ''' xesmf's default sets cells outside the source grid to 0; they must be NaN so
    a smaller, higher-priority region can't overwrite the rest of the mosaic with 0 '''
    small = _source_ds(["sst"], [100.0], lat_range=(LAT_0, 66, 1), lon_range=SOURCE_LON)
    full = _source_ds(["sst"], [200.0])

    alone = _build_pipeline(["sst"])(ds_list=[small], time_mask=np.array([True]))
    values = alone["sst"].values
    assert np.isnan(values).any() and not np.any(values == 0)

    mosaic = _build_pipeline(["sst"])(ds_list=[small, full], time_mask=np.array([True]))
    values = mosaic["sst"].values
    assert np.any(np.isclose(values, 100.0)) and np.any(np.isclose(values, 200.0))
    assert not np.isnan(values).any()


def test_call_rotates_pair_vars_after_regridding():
    target_grid = _target_grid()
    pipeline = _build_pipeline(["u", "v"], pair_vars_list=[("u", "v")], target_grid=target_grid)

    ds = _source_ds(["u", "v"], values_by_time=[1.0])
    ds["v"] = xr.zeros_like(ds["v"])  # pure-u field: (u, v) = (1, 0) everywhere

    result = pipeline(ds_list=[ds], time_mask=np.array([True]))

    assert np.allclose(result["u"].values, target_grid["cos_g"].values)
    assert np.allclose(result["v"].values, target_grid["sin_g"].values)
