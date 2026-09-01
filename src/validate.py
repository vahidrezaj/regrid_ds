'''
[--This module was written by Claude Code--]
Post-run sanity checks for a dataset's saved output.

Validates the Zarr store (time-series datasets) or `.npz` file (static datasets,
e.g. bathymetry) against the current config -- structure, completeness, and basic
data sanity -- without touching the source NetCDF/GeoTIFF files.
'''

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from grid_interp import create_local_metric_grid
from hbm_regridder import _to_plain

logger = logging.getLogger(__name__)


def _store_names(variable_names, variable_attrs):
    ''' source variable name -> name used in the output, per variable_attrs (see ZarrDataWriter) '''
    variable_attrs = variable_attrs or {}
    return [variable_attrs.get(var, {}).get("name", var) for var in variable_names]


def _ok(passed, msg, *args):
    (logger.info if passed else logger.error)(("  ok   " if passed else "  FAIL ") + msg, *args)
    return passed


def _warn(msg, *args):
    logger.warning("  warn " + msg, *args)


def _target_grid(cfg):
    return create_local_metric_grid(
        domain_size_km=cfg.domain.domain_size,
        grid_size=cfg.domain.grid_size,
        lat_0=cfg.domain.lat_0,
        lon_0=cfg.domain.lon_0,
        proj_type="aeqd",
    )


def _sample_time_indices(n_time, n_samples=5):
    ''' a handful of spread-out time indices, cheap to load for a NaN-fraction spot check '''
    if n_time <= n_samples:
        return list(range(n_time))
    return sorted(set(np.linspace(0, n_time - 1, n_samples).round().astype(int).tolist()))


def validate_zarr(cfg, out_path: Path) -> bool:
    ''' validate a time-series dataset's Zarr store against `cfg` '''
    name = cfg.dataset.name
    zarr_path = out_path / f"{name}.zarr"
    cp_path = out_path / f"checkpoint_{name}.tmp"
    variable_names = list(cfg.dataset.variable_names)
    variable_attrs = _to_plain(cfg.dataset.get("variable_attrs", None))
    expected_vars = _store_names(variable_names, variable_attrs)

    ok = _ok(zarr_path.exists(), "store exists: %s", zarr_path)
    if not ok:
        return False

    ok &= _ok(not cp_path.exists(), "run marked complete (no checkpoint file)")
    for part in out_path.glob("*.part"):
        _warn("leftover partial-write file: %s", part)

    expected_time = np.arange(
        np.datetime64(cfg.domain.from_to[0]),
        np.datetime64(cfg.domain.from_to[1]),
        np.timedelta64(cfg.domain.ts, "h"),
    )
    grid_size = cfg.domain.grid_size
    time_chunk = cfg.domain.time_chunk

    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        ok &= _ok(
            ds.sizes.get("time") == len(expected_time)
            and ds.sizes.get("y") == grid_size
            and ds.sizes.get("x") == grid_size,
            "dims match config: time=%d, y=%d, x=%d",
            ds.sizes.get("time"), ds.sizes.get("y"), ds.sizes.get("x"),
        )

        ok &= _ok(
            "time" in ds.coords and np.array_equal(ds["time"].values, expected_time),
            "time axis matches config (%s -> %s, %d steps)",
            cfg.domain.from_to[0], cfg.domain.from_to[1], len(expected_time),
        )

        missing = set(expected_vars) - set(ds.data_vars)
        extra = set(ds.data_vars) - set(expected_vars) - {"nan_mask"}
        ok &= _ok(not missing, "no missing variables (expected %s)", expected_vars)
        if extra:
            _warn("unexpected extra variable(s) in store: %s", sorted(extra))

        ok &= _ok("nan_mask" in ds.data_vars, "nan_mask present")
        if "nan_mask" in ds.data_vars:
            unwritten = np.where(ds["nan_mask"].values)[0]
            ok &= _ok(
                len(unwritten) == 0,
                "all %d timestamps written (0 unwritten)", len(expected_time),
            )
            if len(unwritten):
                logger.error(
                    "       %d/%d timestamps never written, e.g. %s",
                    len(unwritten), len(expected_time),
                    ds["time"].values[unwritten[:5]],
                )

        for var in expected_vars:
            if var not in ds.data_vars:
                continue
            da = ds[var]
            ok &= _ok(
                da.dtype == np.float32 and da.dims == ("time", "y", "x"),
                "%s: dtype/dims correct (%s, %s)", var, da.dtype, da.dims,
            )
            chunksize = getattr(da.data, "chunksize", None)
            if chunksize is not None:
                ok &= _ok(
                    chunksize == (time_chunk, grid_size, grid_size),
                    "%s: chunk shape %s matches config (time_chunk=%d)",
                    var, chunksize, time_chunk,
                )

        if "spatial_ref" in ds.coords:
            crs_attrs = ds["spatial_ref"].attrs
            ok &= _ok(
                np.isclose(crs_attrs.get("latitude_of_projection_origin", np.nan),
                           cfg.domain.lat_0)
                and np.isclose(crs_attrs.get("longitude_of_projection_origin", np.nan),
                                cfg.domain.lon_0),
                "CRS origin matches config (lat_0=%s, lon_0=%s)",
                cfg.domain.lat_0, cfg.domain.lon_0,
            )
        else:
            ok = _ok(False, "spatial_ref coordinate present") and ok

        target_grid = _target_grid(cfg)
        if "lat" in ds.coords and "lon" in ds.coords:
            ok &= _ok(
                np.allclose(ds["lat"].values, target_grid["lat"], atol=1e-6)
                and np.allclose(ds["lon"].values, target_grid["lon"], atol=1e-6),
                "lat/lon grid matches what the current config would generate",
            )
            ok &= _ok(
                bool(np.isfinite(ds["lat"].values).all())
                and bool(np.isfinite(ds["lon"].values).all()),
                "lat/lon fully finite (no NaN/inf)",
            )

        n_time = ds.sizes.get("time", 0)
        for idx in _sample_time_indices(n_time):
            step = ds.isel(time=idx).compute()
            for var in expected_vars:
                if var not in step:
                    continue
                arr = step[var].values
                nan_frac = float(np.isnan(arr).mean())
                if nan_frac >= 1.0:
                    ok = _ok(False, "%s: 100%% NaN at time index %d", var, idx) and ok
                finite = arr[np.isfinite(arr)]
                logger.info(
                    "  info %s @ t=%d: nan=%.1f%%%s",
                    var, idx, nan_frac * 100,
                    f", min={finite.min():.4g}, max={finite.max():.4g}" if finite.size else "",
                )

    return ok


def validate_npz(cfg, out_path: Path) -> bool:
    ''' validate a static dataset's `.npz` output against `cfg` '''
    name = cfg.dataset.name
    npz_path = out_path / f"{name}.npz"
    variable_names = list(cfg.dataset.variable_names)
    variable_attrs = _to_plain(cfg.dataset.get("variable_attrs", None))
    expected_vars = _store_names(variable_names, variable_attrs)

    ok = _ok(npz_path.exists(), "file exists: %s", npz_path)
    if not ok:
        return False

    for part in out_path.glob("*.part"):
        _warn("leftover partial-write file: %s", part)

    grid_size = cfg.domain.grid_size
    expected_keys = set(expected_vars) | {"lat", "lon", "y", "x", "crs"}

    with np.load(npz_path, allow_pickle=True) as payload:
        missing = expected_keys - set(payload.files)
        extra = set(payload.files) - expected_keys
        ok &= _ok(not missing, "no missing arrays (expected %s)", sorted(expected_keys))
        if extra:
            _warn("unexpected extra array(s) in file: %s", sorted(extra))

        for key in ("lat", "lon"):
            if key in payload:
                arr = np.asarray(payload[key])
                ok &= _ok(
                    arr.shape == (grid_size, grid_size),
                    "%s: shape %s matches grid_size=%d", key, arr.shape, grid_size,
                )
        for key in ("y", "x"):
            if key in payload:
                arr = np.asarray(payload[key])
                ok &= _ok(
                    arr.shape == (grid_size,),
                    "%s: shape %s matches grid_size=%d", key, arr.shape, grid_size,
                )

        target_grid = _target_grid(cfg)
        if "lat" in payload and "lon" in payload:
            ok &= _ok(
                np.allclose(payload["lat"], target_grid["lat"], atol=1e-6)
                and np.allclose(payload["lon"], target_grid["lon"], atol=1e-6),
                "lat/lon grid matches what the current config would generate",
            )

        if "crs" in payload:
            crs = np.asarray(payload["crs"]).item()
            ok &= _ok(
                np.isclose(crs.get("latitude_of_projection_origin", np.nan),
                           cfg.domain.lat_0)
                and np.isclose(crs.get("longitude_of_projection_origin", np.nan),
                                cfg.domain.lon_0),
                "CRS origin matches config (lat_0=%s, lon_0=%s)",
                cfg.domain.lat_0, cfg.domain.lon_0,
            )
        else:
            ok = _ok(False, "crs array present") and ok

        for var in expected_vars:
            if var not in payload:
                continue
            arr = np.asarray(payload[var])
            ok &= _ok(
                arr.shape == (grid_size, grid_size),
                "%s: shape %s matches grid_size=%d", var, arr.shape, grid_size,
            )
            nan_frac = float(np.isnan(arr).mean())
            ok = _ok(nan_frac < 1.0, "%s: not 100%% NaN", var) and ok
            finite = arr[np.isfinite(arr)]
            logger.info(
                "  info %s: nan=%.1f%%%s",
                var, nan_frac * 100,
                f", min={finite.min():.4g}, max={finite.max():.4g}" if finite.size else "",
            )

    return ok


def validate_output(cfg) -> bool:
    ''' check the given dataset's saved output on disk for structural correctness and
    completeness against `cfg`. Returns True iff every check passed. '''
    name = cfg.dataset.name
    out_path = Path(cfg.output_path)
    static = bool(cfg.dataset.get("static", False))

    logger.info("[%s] checking output in %s", name, out_path)
    ok = validate_npz(cfg, out_path) if static else validate_zarr(cfg, out_path)

    logger.info("[%s] %s", name, "PASS: output is correct and clean" if ok else "FAIL: see above")
    return ok
