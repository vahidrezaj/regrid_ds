'''Tests for regridder.PreProcessing's file-queue discovery (domain.file_match glob)'''

from omegaconf import OmegaConf

from regridder import PreProcessing


def _make_cfg(folder, tokens, out_path):
    return OmegaConf.create({
        "dataset": {
            "name": "test_ds",
            "folder": str(folder),
            "variable_names": ["sst"],
            "variable_attrs": None,
            "interp_method": "bilinear",
            "extrap_method": None,
            "reader_fn": {"_target_": "io_functions.read_nc", "_partial_": True},
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
