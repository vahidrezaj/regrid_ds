'''Tests for ZarrDataWriter and save_static_npz'''

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from writers import ZarrDataWriter, save_static_npz

# mirrors the literal description writers.py::_initialize attaches to alpha_ref
ALPHA_REF_DESCRIPTION = (
    "Clockwise rotation (in degrees) of the grid's local +y axis from true north "
    "(see alpha_deg in grid_interp.py). Only present when the grid is rotated; its "
    "absence means +y aligns with true north. Note: spatial_ref describes the "
    "unrotated projection and does not account for this rotation. Always use the "
    "explicit 2D lat/lon arrays for geolocation."
)


@pytest.fixture
def target_grid():
    h, w = 4, 4
    y = np.linspace(-1000.0, 1000.0, h)
    x = np.linspace(-1000.0, 1000.0, w)
    lon, lat = np.meshgrid(np.linspace(20.0, 21.0, w), np.linspace(59.0, 60.0, h))
    proj_crs = CRS.from_proj4("+proj=aeqd +lat_0=59.5 +lon_0=20.5 +datum=WGS84 +units=m")
    return {
        "lat": lat,
        "lon": lon,
        "y": y,
        "x": x,
        "crs": proj_crs.to_cf(),
    }


@pytest.fixture
def time_vector():
    return np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2020-01-01T10"),
        np.timedelta64(1, "h"),
    )


def _make_chunk(times, variable_names, grid, fill_value, var_attrs=None):
    h, w = grid["lat"].shape
    var_attrs = var_attrs or {}
    data_vars = {
        var: (
            ("time", "y", "x"),
            np.full((len(times), h, w), fill_value, dtype=np.float32),
            var_attrs.get(var, {}),
        )
        for var in variable_names
    }
    return xr.Dataset(data_vars, coords={"time": times})


def test_write_and_gaps(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst", "ssh"]

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )

    # write two non-contiguous chunks, leaving a gap in between
    chunk_a_times = time_vector[0:3]
    chunk_b_times = time_vector[6:9]
    writer.write(_make_chunk(chunk_a_times, variable_names, target_grid, 1.0))
    writer.write(_make_chunk(chunk_b_times, variable_names, target_grid, 2.0))
    writer.close()

    ds = xr.open_zarr(str(zarr_path), consolidated=True)

    written_mask = np.zeros(len(time_vector), dtype=bool)
    written_mask[0:3] = True
    written_mask[6:9] = True

    for var in variable_names:
        values = ds[var].values
        assert np.all(values[0:3] == 1.0)
        assert np.all(values[6:9] == 2.0)
        assert np.all(np.isnan(values[~written_mask]))

    missing_mask = ds["missing_mask"].values
    assert np.array_equal(missing_mask, ~written_mask)

    # CF grid-mapping metadata present and parseable
    assert "spatial_ref" in ds.coords
    crs = CRS.from_cf(ds["spatial_ref"].attrs)
    assert crs.to_cf()["grid_mapping_name"] == "azimuthal_equidistant"
    assert ds["sst"].attrs["grid_mapping"] == "spatial_ref"

    # alpha_ref (grid rotation) is only saved when the grid is actually rotated --
    # target_grid here has no "alpha_deg" key (defaults to 0.0), so it must be absent
    assert "alpha_ref" not in ds.coords

    ds.close()


def test_write_preserves_variable_attrs(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst"]

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )

    # source attrs include a "grid_mapping" left over from the original
    # NetCDF file -- must not overwrite the one we set ourselves.
    source_attrs = {
        "sst": {
            "units": "degC",
            "long_name": "sea surface temperature",
            "grid_mapping": "some_other_crs_var",
        }
    }
    writer.write(_make_chunk(time_vector[0:2], variable_names, target_grid, 1.0, source_attrs))
    writer.close()

    ds = xr.open_zarr(str(zarr_path), consolidated=True)
    assert ds["sst"].attrs["units"] == "degC"
    assert ds["sst"].attrs["long_name"] == "sea surface temperature"
    assert ds["sst"].attrs["grid_mapping"] == "spatial_ref"
    ds.close()


def test_write_rejects_unknown_time(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst"]

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )

    # after the end of time_vector: searchsorted returns an out-of-bounds index
    after_range = np.array([np.datetime64("2020-01-02T00")])
    with pytest.raises(ValueError):
        writer.write(_make_chunk(after_range, variable_names, target_grid, 1.0))

    # before the start of time_vector: searchsorted returns a valid (but wrong) index
    before_range = np.array([np.datetime64("2019-12-31T23")])
    with pytest.raises(ValueError):
        writer.write(_make_chunk(before_range, variable_names, target_grid, 1.0))

    writer.close()


def test_write_rejects_missing_variable(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst", "ssh"]

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )

    with pytest.raises(ValueError):
        writer.write(_make_chunk(time_vector[0:1], ["sst"], target_grid, 1.0))
    writer.close()


def test_reopen_validates_configuration(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst"]

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )
    writer.close()

    # reopening with the same config should succeed
    writer2 = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        time_chunk=4,
    )
    writer2.close()

    # reopening with a mismatched time_vector should raise
    other_time_vector = time_vector[:-1]
    with pytest.raises(ValueError):
        ZarrDataWriter(
            zarr_path=str(zarr_path),
            time_vector=other_time_vector,
            variable_names=variable_names,
            target_grid=target_grid,
            time_chunk=4,
        )

    # reopening with a different CRS origin (same shape) should raise
    other_grid = dict(target_grid)
    other_grid["crs"] = CRS.from_proj4(
        "+proj=aeqd +lat_0=10.0 +lon_0=10.0 +datum=WGS84 +units=m"
    ).to_cf()
    with pytest.raises(ValueError):
        ZarrDataWriter(
            zarr_path=str(zarr_path),
            time_vector=time_vector,
            variable_names=variable_names,
            target_grid=other_grid,
            time_chunk=4,
        )

    # reopening with the same CRS origin but a different lat/lon grid (e.g. alpha_deg
    # changed between runs, which CRS origin alone can't detect -- see grid_interp.py)
    # should also raise
    rotated_grid = dict(target_grid)
    rotated_grid["lat"] = target_grid["lat"] + 5.0
    with pytest.raises(ValueError):
        ZarrDataWriter(
            zarr_path=str(zarr_path),
            time_vector=time_vector,
            variable_names=variable_names,
            target_grid=rotated_grid,
            time_chunk=4,
        )


def test_alpha_ref_saved_only_when_rotated(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst"]

    rotated_grid = dict(target_grid)
    rotated_grid["alpha_deg"] = 12.5
    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=rotated_grid,
        time_chunk=4,
    )
    writer.close()

    ds = xr.open_zarr(str(zarr_path), consolidated=True)
    assert "alpha_ref" in ds.coords
    assert float(ds["alpha_ref"].values) == 12.5
    assert ds["alpha_ref"].attrs["description"] == ALPHA_REF_DESCRIPTION
    ds.close()


def test_reopen_detects_alpha_deg_mismatch(tmp_path, target_grid, time_vector):
    ''' alpha_deg mismatch gets its own specific error message, distinct from the generic
    lat/lon-mismatch fallback (both ultimately guard against the same kind of drift) '''
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["sst"]

    rotated_grid = dict(target_grid)
    rotated_grid["alpha_deg"] = 12.5
    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=rotated_grid,
        time_chunk=4,
    )
    writer.close()

    # target_grid itself has no "alpha_deg" key -> defaults to 0.0, which mismatches
    # the 12.5 stored above
    with pytest.raises(ValueError, match="alpha_deg"):
        ZarrDataWriter(
            zarr_path=str(zarr_path),
            time_vector=time_vector,
            variable_names=variable_names,
            target_grid=target_grid,
            time_chunk=4,
        )


def test_write_renames_and_overrides_attrs(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["var33", "var34"]
    variable_attrs = {
        "var33": {"name": "uwind", "units": "m/s"},
        "var34": {"name": "vwind", "units": "m/s"},
    }

    writer = ZarrDataWriter(
        zarr_path=str(zarr_path),
        time_vector=time_vector,
        variable_names=variable_names,
        target_grid=target_grid,
        variable_attrs=variable_attrs,
        time_chunk=4,
    )

    # source has no attrs at all, as with raw cdo-converted GRIB output;
    # a stray "table" attr (real GRIB metadata) must not leak into the store
    source_attrs = {"var33": {"table": 1}, "var34": {"table": 1}}
    writer.write(_make_chunk(time_vector[0:2], variable_names, target_grid, 1.0, source_attrs))
    writer.close()

    ds = xr.open_zarr(str(zarr_path), consolidated=True)
    assert "var33" not in ds.data_vars
    assert "var34" not in ds.data_vars
    assert ds["uwind"].attrs["units"] == "m/s"
    assert ds["vwind"].attrs["units"] == "m/s"
    assert "table" not in ds["uwind"].attrs
    assert "name" not in ds["uwind"].attrs
    assert np.all(ds["uwind"].values[0:2] == 1.0)
    ds.close()


def test_variable_attrs_rejects_duplicate_names(tmp_path, target_grid, time_vector):
    zarr_path = tmp_path / "test.zarr"
    variable_names = ["var33", "var34"]
    variable_attrs = {
        "var33": {"name": "wind", "units": "m/s"},
        "var34": {"name": "wind", "units": "m/s"},
    }

    with pytest.raises(ValueError):
        ZarrDataWriter(
            zarr_path=str(zarr_path),
            time_vector=time_vector,
            variable_names=variable_names,
            target_grid=target_grid,
            variable_attrs=variable_attrs,
            time_chunk=4,
        )


def test_save_static_npz(tmp_path, target_grid):
    npz_path = tmp_path / "hbm_bathymetry.npz"
    level = np.arange(target_grid["lat"].size, dtype=np.float32).reshape(target_grid["lat"].shape)

    save_static_npz(npz_path, {"level": level}, target_grid)

    assert npz_path.exists()
    payload = np.load(npz_path, allow_pickle=True)
    assert np.array_equal(payload["level"], level)
    assert np.allclose(payload["lat"], target_grid["lat"])
    assert np.allclose(payload["lon"], target_grid["lon"])
    assert payload["crs"].item() == target_grid["crs"]
    # target_grid fixture has no "alpha_deg" key -> alpha_ref must be absent, not 0.0
    assert "alpha_ref" not in payload.files


def test_save_static_npz_includes_alpha_ref_when_rotated(tmp_path, target_grid):
    npz_path = tmp_path / "nemo.npz"
    level = np.arange(target_grid["lat"].size, dtype=np.float32).reshape(target_grid["lat"].shape)
    rotated_grid = dict(target_grid)
    rotated_grid["alpha_deg"] = 12.5

    save_static_npz(npz_path, {"level": level}, rotated_grid)

    payload = np.load(npz_path, allow_pickle=True)
    assert payload["alpha_ref"] == 12.5
