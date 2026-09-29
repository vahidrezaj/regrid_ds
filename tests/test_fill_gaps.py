'''Tests for fill_gaps.fill_time_gaps'''

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from fill_gaps import _find_gaps, fill_time_gaps
from writers import ZarrDataWriter


@pytest.fixture
def target_grid():
    h, w = 3, 4
    lon, lat = np.meshgrid(np.linspace(20.0, 21.0, w), np.linspace(59.0, 60.0, h))
    proj_crs = CRS.from_proj4("+proj=aeqd +lat_0=59.5 +lon_0=20.5 +datum=WGS84 +units=m")
    return {
        "lat": lat,
        "lon": lon,
        "y": np.linspace(-1000.0, 1000.0, h),
        "x": np.linspace(-1000.0, 1000.0, w),
        "crs": proj_crs.to_cf(),
    }


@pytest.fixture
def time_vector():
    return np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2020-01-01T20"),
        np.timedelta64(1, "h"),
    )


def _signal(n_time, grid):
    ''' linear in time, different per cell; cell (0, 0) is land (NaN) '''
    h, w = grid["lat"].shape
    t = np.arange(n_time, dtype=np.float32)[:, None, None]
    data = t * 2.0 + np.arange(h * w, dtype=np.float32).reshape(1, h, w)
    data[:, 0, 0] = np.nan
    return data


def _make_store(path, time_vector, grid, written, **writer_kw):
    ''' store holding `_signal` at the `written` steps only '''
    data = _signal(len(time_vector), grid)
    writer = ZarrDataWriter(
        zarr_path=str(path),
        time_vector=time_vector,
        variable_names=["sst", "ssh"],
        target_grid=grid,
        **writer_kw,
    )
    idx = np.flatnonzero(written)
    writer.write(xr.Dataset(
        {v: (("time", "y", "x"), data[idx]) for v in ("sst", "ssh")},
        coords={"time": time_vector[idx]},
    ))
    return data


def test_find_gaps():
    missing = np.array([1, 0, 0, 1, 1, 0, 1], dtype=bool)
    assert _find_gaps(missing) == [(0, 1), (3, 5), (6, 7)]
    assert _find_gaps(np.zeros(3, dtype=bool)) == []


@pytest.mark.parametrize("writer_kw", [{"time_chunk": 4}, {"time_chunk": 1, "time_shard": 4}])
def test_fill_short_gaps(tmp_path, target_grid, time_vector, writer_kw):
    zarr_path = tmp_path / "test.zarr"
    written = np.ones(len(time_vector), dtype=bool)
    written[[0, 1]] = False          # at start: skipped
    written[[4]] = False             # length 1: filled
    written[[7, 8]] = False          # length 2, crosses a shard boundary: filled
    written[[11, 12, 13]] = False    # length 3 > gap_len: skipped
    written[[19]] = False            # at end: skipped
    data = _make_store(zarr_path, time_vector, target_grid, written, **writer_kw)

    summary = fill_time_gaps(zarr_path, gap_len=2)
    assert summary == {"gaps_filled": 2, "steps_filled": 3, "gaps_too_long": 1, "gaps_at_edge": 2}

    filled = np.zeros(len(time_vector), dtype=bool)
    filled[[4, 7, 8]] = True
    still_missing = ~written & ~filled

    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        for var in ("sst", "ssh"):
            values = ds[var].values
            # signal is linear in time, so interpolation is exact
            np.testing.assert_allclose(values[written | filled], data[written | filled], rtol=1e-6)
            assert np.all(np.isnan(values[still_missing]))
            assert np.all(np.isnan(values[:, 0, 0]))
        np.testing.assert_array_equal(ds["missing_mask"].values, still_missing)
        np.testing.assert_array_equal(ds["interp_mask"].values, filled)
        assert ds["interp_mask"].dims == ("time",)

    # second run: nothing left that fits
    assert fill_time_gaps(zarr_path, gap_len=2)["gaps_filled"] == 0


def test_weights_follow_time_not_index(tmp_path, target_grid):
    ''' uneven time axis: weights come from the actual timestamps '''
    time_vector = np.array(
        ["2020-01-01T00", "2020-01-01T01", "2020-01-01T04", "2020-01-01T05"],
        dtype="datetime64[ns]",
    )
    zarr_path = tmp_path / "test.zarr"
    written = np.array([True, False, False, True])
    writer = ZarrDataWriter(str(zarr_path), time_vector, ["sst"], target_grid, time_chunk=4)
    h, w = target_grid["lat"].shape
    writer.write(xr.Dataset(
        {"sst": (("time", "y", "x"), np.stack([np.zeros((h, w)), np.full((h, w), 5.0)]))},
        coords={"time": time_vector[written]},
    ))

    fill_time_gaps(zarr_path, gap_len=2)
    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        np.testing.assert_allclose(ds["sst"].values[:, 0, 0], [0.0, 1.0, 4.0, 5.0])


def test_dry_run_writes_nothing(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    written = np.ones(len(time_vector), dtype=bool)
    written[5] = False
    _make_store(zarr_path, time_vector, target_grid, written, time_chunk=4)

    summary = fill_time_gaps(zarr_path, gap_len=1, dry_run=True)
    assert summary["gaps_filled"] == 1

    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        assert "interp_mask" not in ds
        assert ds["missing_mask"].values[5]
        assert np.all(np.isnan(ds["sst"].values[5]))


def test_rejects_bad_gap_len(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    _make_store(zarr_path, time_vector, target_grid, np.ones(len(time_vector), bool),
                time_chunk=4)
    with pytest.raises(ValueError, match="gap_len"):
        fill_time_gaps(zarr_path, gap_len=0)
