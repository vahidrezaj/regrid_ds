'''Tests for regridder.PreProcessing: file-queue discovery (domain.file_match glob),
file_batch batching + checkpoint/resume, and the land/valid masks'''

import json

import numpy as np
import pytest
import xarray as xr
import zarr
from omegaconf import OmegaConf

from grid_interp import create_local_metric_grid
from land_mask import LAND_MASK_FILE
from regridder import PreProcessing
from writers import save_static_npz


def _make_cfg(folder, tokens, out_path):
    return OmegaConf.create({
        "dataset": {
            "name": "test_ds",
            "source": "test",
            "folder": str(folder),
            "variable_names": ["sst"],
            "variable_attrs": None,
            "interp_method": "bilinear",
            "reader_fn": {"_target_": "readers.read_nc", "_partial_": True},
        },
        "domain": {
            "file_match": {"test_ds": tokens},
            "domain_size": 100,
            "grid_size": 3,
            "lat_0": 60.0,
            "lon_0": 10.0,
            "from_to": ["2000-01-01T00:00", "2000-01-02T00:00"],
            "ts": 1,
            "time_chunk": 24,
            "clevel": 3,
        },
        "output_path": str(out_path),
        "mode": "dry_run",
        "verbose": False,
    })


def test_glob_matches_leading_prefix_filenames(tmp_path):
    ''' HBM-style: token at the start of the filename (e.g. "IDW_20200101.nc") '''
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "IDW_20200101.nc").touch()
    (data_dir / "other_20200101.nc").touch()

    cfg = _make_cfg(data_dir, ["IDW"], tmp_path / "out")
    pp = PreProcessing(cfg, base_path="")

    assert pp.total_files_all == 1


def test_glob_handles_null_file_match(tmp_path):
    ''' domain.file_match: null (or missing) -> tokens=[""], which must not turn
    into the "**" + ext pattern (pathlib rejects "**" outside its own path component) '''
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "anything_20200101.nc").touch()
    (data_dir / "other_20200102.nc").touch()

    cfg = _make_cfg(data_dir, None, tmp_path / "out")
    pp = PreProcessing(cfg, base_path="")

    assert pp.total_files_all == 2


def test_glob_matches_trailing_suffix_filenames(tmp_path):
    ''' NEMO-style: token as a suffix before the extension
    (e.g. "NAA10KM_1h_20200101_20201231_ssh.nc") -- the case that motivated
    generalizing the glob from a leading-prefix-only pattern. '''
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "NAA10KM_1h_20200101_20201231_ssh.nc").touch()
    (data_dir / "NAA10KM_1h_20200101_20201231_ubar.nc").touch()

    cfg = _make_cfg(data_dir, ["ssh"], tmp_path / "out")
    pp = PreProcessing(cfg, base_path="")

    assert pp.total_files_all == 1


N_HOURS = 10


def _write_hourly_files(data_dir):
    ''' one 1-step file per hour (hbm_forcing-style), values unique per hour '''
    data_dir.mkdir()
    lat = np.linspace(59.0, 61.0, 9)
    lon = np.linspace(8.0, 12.0, 9)
    for h in range(N_HOURS):
        time = np.array([np.datetime64("2000-01-01T00:00") + np.timedelta64(h, "h")])
        sst = h + np.add.outer(lat, lon)[None].astype("float32")
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst)},
            coords={"time": time, "lat": lat, "lon": lon},
        ).to_netcdf(data_dir / f"20000101{h:02d}.nc")


def _make_run_cfg(data_dir, out_path, file_batch, time_batch):
    cfg = _make_cfg(data_dir, None, out_path)
    cfg.mode = "run"
    cfg.dataset.fill_method = "nearest"
    cfg.dataset.file_batch = file_batch
    cfg.domain.from_to = ["2000-01-01T00:00", "2000-01-01T12:00"]
    cfg.domain.time_chunk = 1
    cfg.domain.time_shard = 4
    cfg.domain.time_batch = time_batch
    return cfg


def _read_store(out_path):
    store = zarr.open_group(out_path / "ds.zarr", mode="r")
    return np.asarray(store["sst"][:]), np.asarray(store["missing_mask"][:], dtype=bool)


@pytest.fixture
def hourly_dir(tmp_path):
    data_dir = tmp_path / "data"
    _write_hourly_files(data_dir)
    return data_dir


@pytest.fixture
def reference(hourly_dir, tmp_path):
    ''' store written one file at a time (file_batch=1) '''
    out = tmp_path / "ref"
    PreProcessing(_make_run_cfg(hourly_dir, out, file_batch=1, time_batch=None))()
    return _read_store(out)


@pytest.mark.parametrize("file_batch, time_batch", [(3, None), (3, 2), (4, 4), (50, None)])
def test_file_batch_matches_per_file_run(hourly_dir, tmp_path, reference, file_batch, time_batch):
    ''' batching files (incl. a short last batch, and time_batch splitting a batch) must
    write exactly what the per-file loop writes '''
    out = tmp_path / "batched"
    pp = PreProcessing(_make_run_cfg(hourly_dir, out, file_batch, time_batch))
    pp()

    sst, missing = _read_store(out)
    ref_sst, ref_missing = reference
    np.testing.assert_array_equal(missing, ref_missing)
    assert (~missing).sum() == N_HOURS
    np.testing.assert_allclose(sst, ref_sst, equal_nan=True)
    assert not pp.cp_path.exists()


def test_file_batch_checkpoint_resume_after_crash_mid_batch(hourly_dir, tmp_path, reference):
    ''' a crash partway through a batch must leave the checkpoint at the start of that
    batch (not past it, not mid-batch), and a fresh instance must resume from there '''
    out = tmp_path / "crash"
    cfg = _make_run_cfg(hourly_dir, out, file_batch=3, time_batch=2)
    pp = PreProcessing(cfg)

    # batch 1 = 2 writes ([00,01], [02]); crash on the 2nd write of batch 2 ([05])
    real_write, calls = pp.writer.write, []
    def failing_write(ds):
        calls.append(ds)
        if len(calls) == 4:
            raise RuntimeError("simulated crash")
        real_write(ds)
    pp.writer.write = failing_write

    with pytest.raises(RuntimeError, match="simulated crash"):
        pp()

    remaining = json.loads(pp.cp_path.read_text(encoding="utf-8"))
    assert [p.split("\\")[-1].split("/")[-1] for p in remaining[0]] == \
        [f"20000101{h:02d}.nc" for h in range(3, N_HOURS)]

    resumed = PreProcessing(cfg)
    assert resumed.total_files == N_HOURS - 3
    resumed()

    sst, missing = _read_store(out)
    ref_sst, ref_missing = reference
    np.testing.assert_array_equal(missing, ref_missing)
    np.testing.assert_allclose(sst, ref_sst, equal_nan=True)
    assert not resumed.cp_path.exists()


def test_run_with_land_mask_saves_land_and_valid_masks(hourly_dir, tmp_path):
    out = tmp_path / "out"
    cfg = _make_run_cfg(hourly_dir, out, file_batch=3, time_batch=None)
    cfg.dataset.land_mask = True
    grid = create_local_metric_grid(100, 3, 60.0, 10.0)
    land = np.zeros(grid["lat"].shape, dtype=bool)
    land[:, 0] = True
    save_static_npz(out / LAND_MASK_FILE, {"land_mask": land}, grid)

    PreProcessing(cfg)()

    store = zarr.open_group(out / "ds.zarr", mode="r")
    np.testing.assert_array_equal(store["land_mask"][:], land)
    np.testing.assert_array_equal(store["valid_mask"][:], ~land)  # source covers the grid
    sst, missing = _read_store(out)
    written = sst[~missing]
    assert np.isnan(written[:, land]).all() and np.isfinite(written[:, ~land]).all()


def test_run_without_land_mask_file_fails_early(hourly_dir, tmp_path):
    cfg = _make_run_cfg(hourly_dir, tmp_path / "out", file_batch=1, time_batch=None)
    cfg.dataset.land_mask = True
    with pytest.raises(FileNotFoundError, match="mode=land_mask"):
        PreProcessing(cfg)
    assert not (tmp_path / "out" / "ds.zarr").exists()


@pytest.mark.parametrize("file_batch", [None, 0, -5])
def test_file_batch_falls_back_to_default(hourly_dir, tmp_path, file_batch):
    cfg = _make_run_cfg(hourly_dir, tmp_path / "out", file_batch=file_batch, time_batch=None)
    cfg.mode = "dry_run"
    assert PreProcessing(cfg).file_batch == 1
