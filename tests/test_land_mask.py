'''Tests for land_mask: build_land_mask (model land inside coverage, reference land
outside it) and load_land_mask'''

import os

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

import land_mask
from grid_interp import create_local_metric_grid

LAT_0, LON_0 = 60.0, 10.0


def _cfg(data_dir, out_path):
    return OmegaConf.create({
        "dataset": {
            "name": "test_ocean",
            "source": "test",
            "folder": str(data_dir),
            "variable_names": ["sst"],
            "reader_fn": {"_target_": "readers.read_nc", "_partial_": True},
        },
        "domain": {
            "name": "test_domain",
            "file_match": {"test_ocean": None},
            "domain_size": 400,
            "grid_size": 21,
            "lat_0": LAT_0,
            "lon_0": LON_0,
        },
        "output_path": str(out_path),
    })


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ''' source covering only the target grid's southern half. Model NaN: land west of LON_0
    (reference agrees), a dead strip along the source's northern edge (past its open
    boundary: reference says sea) and an islet the reference also calls sea '''
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lat = np.linspace(57.0, LAT_0 + 0.2, 40)
    lon = np.linspace(4.0, 16.0, 60)
    lon2d, lat2d = np.meshgrid(lon, lat)
    model_nan = (
        (lon2d < LON_0)
        | (lat2d > LAT_0 - 0.4)
        | ((np.abs(lat2d - 58.5) < 0.2) & (np.abs(lon2d - 12) < 0.4))
    )
    sst = np.where(model_nan, np.nan, 10.0)
    xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst[None].astype("float32"))},
        coords={"time": [np.datetime64("2000-01-01")], "lat": lat, "lon": lon},
    ).to_netcdf(data_dir / "2000010100.nc")

    def reference(lat, lon, shapefile=None):
        ''' land west of LON_0, plus a land band north of LAT_0 + 0.8 with a sea pocket in
        it (lat > LAT_0 + 1.2), cut off from the model's sea '''
        band = (lat > LAT_0 + 0.8) & ~((lat > LAT_0 + 1.2) & (np.abs(lon - 12) < 1))
        return (lon < LON_0) | band

    monkeypatch.setattr(land_mask, "reference_land", reference)
    monkeypatch.setattr(land_mask, "_plot", lambda *args: None)

    cfg = _cfg(data_dir, tmp_path / "out")
    grid = create_local_metric_grid(400, 21, LAT_0, LON_0)
    return cfg, grid


def test_build_uses_model_inside_coverage_and_reference_outside(setup):
    cfg, grid = setup
    land_mask.build_land_mask(cfg)

    with np.load(f"{cfg.output_path}/{land_mask.LAND_MASK_FILE}", allow_pickle=True) as npz:
        land, covered = npz["land_mask"], npz["covered"]

    lat, lon = grid["lat"], grid["lon"]
    assert covered.any() and (~covered).any()
    assert not covered[lat > LAT_0 + 0.3].any()  # north of the source: no coverage
    # inside coverage: model's own land west of LON_0
    west = covered & (lon < LON_0 - 0.5)
    assert land[west].all()
    # dead strip past the open boundary (model NaN, reference sea, next to uncovered sea): sea
    strip = covered & (lat > LAT_0 - 0.3) & (lon > LON_0 + 0.5)
    assert strip.any() and not land[strip].any()
    # islet the reference calls sea, but cut off from uncovered sea: model's land kept
    islet = (np.abs(lat - 58.5) < 0.1) & (np.abs(lon - 12) < 0.2)
    assert islet.any() and land[islet].all()
    # outside coverage: reference land, plus the cut-off sea pocket (no data, never filled)
    pocket = (lat > LAT_0 + 1.3) & (np.abs(lon - 12) < 0.8)
    assert pocket.any() and land[pocket].all()
    expected = (lon < LON_0) | (lat > LAT_0 + 0.8)
    np.testing.assert_array_equal(land[~covered], expected[~covered])
    # sea between the source and the band borders the model's sea (via the strip): filled
    to_fill = ~covered & (lat > LAT_0 + 0.3) & (lat < LAT_0 + 0.7) & (lon > LON_0 + 0.5)
    assert to_fill.any() and not land[to_fill].any()


def test_build_skips_existing_and_load_round_trips(setup):
    cfg, grid = setup
    land_mask.build_land_mask(cfg)
    npz_path = f"{cfg.output_path}/{land_mask.LAND_MASK_FILE}"
    mtime = os.path.getmtime(npz_path)

    land_mask.build_land_mask(cfg)  # already there: no rebuild
    assert os.path.getmtime(npz_path) == mtime

    loaded = land_mask.load_land_mask(cfg.output_path, grid)
    assert loaded.dtype == bool and loaded.shape == grid["lat"].shape


def test_load_rejects_missing_file_and_other_grid(setup, tmp_path):
    cfg, grid = setup
    with pytest.raises(FileNotFoundError, match="mode=land_mask"):
        land_mask.load_land_mask(tmp_path / "nowhere", grid)

    land_mask.build_land_mask(cfg)
    other = create_local_metric_grid(400, 21, LAT_0 + 1, LON_0)
    with pytest.raises(ValueError, match="different target grid"):
        land_mask.load_land_mask(cfg.output_path, other)
