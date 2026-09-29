'''
Rechunk an already-saved time-series Zarr store (e.g. `ocean.zarr`, `forcing.zarr`) to the
chunk/shard layout of the current domain config (`domain.time_chunk` / `time_shard` / `clevel`),
i.e. the same layout `ZarrDataWriter` writes now.

The new store is written next to the original (`<name>.rechunk.zarr`), one shard-aligned block
of timesteps at a time. Progress is checkpointed after every block, so an interrupted run resumes
where it stopped. Once written, a few timesteps are compared against the original, then the two
are swapped: the original is kept as `<name>.zarr.bak` -- delete it yourself once you're happy.
Needs free disk space for a full second copy of the store while it runs.

    python run.py mode=rechunk dataset=hbm_ocean domain=baltic_sea
    python run.py -m mode=rechunk dataset=hbm_ocean,hbm_forcing save_to=H:/data_new
'''

import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from time import monotonic

import numpy as np
import xarray as xr
import zarr
from omegaconf import DictConfig
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

# encoding keys describing the old layout (plus `coordinates`, which xarray regenerates on write);
# everything else (dtype, fill value, time units, ...) is carried over unchanged
_LAYOUT_KEYS = {"chunks", "preferred_chunks", "shards", "coordinates"}


def _encoding(src, time_chunk, time_shard, clevel):
    ''' new encoding: (time, y, x) variables get the configured chunks/shards and the same codec
    as `ZarrDataWriter`; 1-D time variables (e.g. missing_mask) and 2-D lat/lon a single chunk '''
    compressor = BloscCodec(cname="zstd", clevel=clevel, shuffle="bitshuffle")

    encoding = {}
    for name, var in src.variables.items():
        enc = {k: v for k, v in var.encoding.items() if k not in _LAYOUT_KEYS}
        if var.dims == ("time", "y", "x"):
            enc["chunks"] = (time_chunk, *var.shape[1:])
            enc["compressors"] = [compressor]
            if time_shard:
                enc["shards"] = (time_shard, *var.shape[1:])
        elif (var.dims == ("time",) and name != "time") or var.dims == ("y", "x"):
            enc["chunks"] = var.shape
        encoding[name] = enc
    return encoding


def _read_cp(cp_path):
    with cp_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_cp(cp_path, next_block):
    ''' atomic checkpoint write, same pattern as regridder.PreProcessing._update_cp '''
    tmp_path = cp_path.with_name(cp_path.name + ".part")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(next_block, f)
    os.replace(tmp_path, cp_path)


def rechunk(src_path: Path, time_chunk, time_shard, clevel):
    ''' rechunk `src_path` into the configured layout, verify, then swap it in place '''
    new_path = src_path.with_suffix(".rechunk.zarr")
    bak_path = src_path.with_suffix(".zarr.bak")
    cp_path = src_path.parent / f"checkpoint_rechunk_{src_path.stem}.tmp"
    file_chunk = time_shard or time_chunk

    src = xr.open_zarr(src_path, consolidated=True)
    grid_vars = [v for v in src.data_vars if src[v].dims == ("time", "y", "x")]
    grid_var = grid_vars[0]
    spatial = src[grid_var].shape[1:]
    target = ((time_chunk, *spatial), (time_shard, *spatial) if time_shard else None)

    ref = src[grid_var].encoding
    if (tuple(ref["chunks"]), tuple(ref["shards"]) if ref.get("shards") else None) == target:
        logger.info("[%s] already chunks=%s, shards=%s -- nothing to do", src_path, *target)
        src.close()
        return
    if bak_path.exists():
        raise FileExistsError(f"{bak_path} already exists -- remove it before rechunking again")

    # resume state: checkpoint = next block to write; a finished new store has no checkpoint
    if cp_path.exists():
        next_block = _read_cp(cp_path)
    elif new_path.exists():
        next_block = None
        logger.info("[%s] %s already fully written, verifying", src_path.name, new_path.name)
    else:
        next_block = 0
        _write_cp(cp_path, 0)

    n_time = src.sizes["time"]
    blocks = [slice(i, min(i + file_chunk, n_time)) for i in range(0, n_time, file_chunk)]

    if next_block == 0:
        # empty store with the new layout; small variables (coords, CRS, missing_mask, ...) are
        # loaded and written right away, the (time, y, x) ones block by block below
        template = src.chunk({"time": file_chunk})
        template = template.assign_coords(
            {c: template[c].compute() for c in template.coords if c not in template.indexes}
        )
        template = template.assign(
            {v: template[v].compute() for v in template.data_vars if v not in grid_vars}
        )
        template.to_zarr(
            new_path, mode="w", compute=False, zarr_format=3,
            encoding=_encoding(src, time_chunk, time_shard, clevel),
        )
        logger.info(
            "[%s] -> %s: chunks=%s, shards=%s, %d blocks of %d steps",
            src_path.name, new_path.name, *target, len(blocks), file_chunk,
        )

    if next_block is not None:
        start = monotonic()
        for i in range(next_block, len(blocks)):
            block = src[grid_vars].isel(time=blocks[i]).chunk({"time": file_chunk})
            block = block.drop_vars([c for c in block.coords if "time" not in block[c].dims])
            block.to_zarr(new_path, region={"time": blocks[i]})
            _write_cp(cp_path, i + 1)

            done = i + 1 - next_block
            eta = (monotonic() - start) / done * (len(blocks) - i - 1)
            logger.info(
                "[%s] block %d/%d (%s -> %s) | ETA %s",
                src_path.name, i + 1, len(blocks),
                src.time.values[blocks[i].start], src.time.values[blocks[i].stop - 1],
                timedelta(seconds=int(eta)),
            )
        zarr.consolidate_metadata(new_path)
        cp_path.unlink()

    # verify: same variables/attrs/coords, and identical values at a few spread-out timesteps
    new = xr.open_zarr(new_path, consolidated=True)
    try:
        for idx in sorted(set(np.linspace(0, n_time - 1, 6).round().astype(int).tolist())):
            xr.testing.assert_identical(src.isel(time=idx).compute(), new.isel(time=idx).compute())
        for v in src.data_vars:
            if src[v].dims == ("time",):                  # e.g. missing_mask: cheap, check all
                xr.testing.assert_identical(src[v].compute(), new[v].compute())
        assert tuple(new[grid_var].encoding["chunks"]) == target[0]
    finally:
        new.close()
        src.close()
    logger.info("[%s] verified against the original", new_path.name)

    # swap: original kept as backup, never deleted here
    src_path.rename(bak_path)
    new_path.rename(src_path)
    logger.info("[%s] done, original kept at %s", src_path, bak_path)


def rechunk_output(cfg: DictConfig):
    ''' rechunk `<output_path>/<dataset>.zarr` for the selected dataset/domain '''
    if cfg.dataset.get("static", False):
        logger.info("[%s] static dataset (.npz), nothing to rechunk", cfg.dataset.name)
        return

    name = cfg.dataset.name.removeprefix(f"{cfg.dataset.source}_")
    rechunk(
        src_path=Path(cfg.output_path) / f"{name}.zarr",
        time_chunk=cfg.domain.time_chunk,
        time_shard=cfg.domain.get("time_shard", None),
        clevel=cfg.domain.clevel,
    )
