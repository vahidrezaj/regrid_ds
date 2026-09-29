'''
Fill short gaps (missing timesteps) in an already-saved time-series Zarr store by linear
interpolation in time between the written steps on either side.

Only runs of at most `gap_len` consecutive missing steps are filled; longer runs and runs at
the start/end of the store (nothing to interpolate from on one side) stay NaN. Filled steps get
`missing_mask=False` and `interp_mask=True`, so they can still be told apart from real data.

    python src/fill_gaps.py data/hbm_baltic_sea/ocean.zarr --gap-len 6 [--dry-run]
    python run.py mode=fill_gaps gap_len=6 dataset=hbm_ocean domain=baltic_sea
'''

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger(__name__)


def _find_gaps(missing):
    ''' (start, stop) of every run of True in `missing` '''
    edges = np.diff(np.concatenate([[0], missing.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_time_gaps(zarr_path, gap_len, dry_run=False):
    '''
    Linearly interpolate every gap of <= `gap_len` missing timesteps in a `ZarrDataWriter` store.

    zarr_path : store to fill, in place
    gap_len : longest run of consecutive missing timesteps to fill
    dry_run : only report what would be filled

    Returns a summary dict (also logged).
    '''
    if gap_len < 1:
        raise ValueError(f"gap_len must be >= 1, got {gap_len}")
    zarr_path = Path(zarr_path)

    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        variables = [v for v in ds.data_vars if ds[v].dims == ("time", "y", "x")]
        if not variables:
            raise ValueError(f"no (time, y, x) variables in {zarr_path}")
        times = ds["time"].values.astype("datetime64[ns]").astype(np.int64)
        missing = ds["missing_mask"].values.astype(bool)
        has_interp_mask = "interp_mask" in ds

        # shard (or chunk) size along time: the unit zarr rewrites on every write
        enc = ds[variables[0]].encoding
        block = (enc.get("shards") or enc["chunks"])[0]

    n_time = len(times)
    to_fill, too_long, at_edge = [], [], []
    for start, stop in _find_gaps(missing):
        if start == 0 or stop == n_time:
            at_edge.append((start, stop))
        elif stop - start > gap_len:
            too_long.append((start, stop))
        else:
            to_fill.append((start, stop))

    summary = {
        "gaps_filled": len(to_fill),
        "steps_filled": int(sum(stop - start for start, stop in to_fill)),
        "gaps_too_long": len(too_long),
        "gaps_at_edge": len(at_edge),
    }
    logger.info(
        "[%s] %s%d gaps (%d steps) to fill, %d longer than gap_len=%d, %d at start/end",
        zarr_path.name, "dry run: " if dry_run else "",
        summary["gaps_filled"], summary["steps_filled"],
        summary["gaps_too_long"], gap_len, summary["gaps_at_edge"],
    )
    if dry_run or not to_fill:
        return summary

    if not has_interp_mask:
        xr.Dataset({
            "interp_mask": ("time", np.zeros(n_time, dtype=bool), {
                "long_name": "time-interpolated mask",
                "description": "True = timestep was missing and filled by linear interpolation "
                               "in time (see fill_gaps.py)",
            }),
        }).to_zarr(
            zarr_path, mode="a", consolidated=True,
            encoding={"interp_mask": {"chunks": (n_time,)}},
        )
    store = zarr.open_group(zarr_path, mode="a")

    # group gaps by the shard their first step falls in, so each shard is rewritten ~once
    batches = {}
    for start, stop in to_fill:
        batches.setdefault(start // block, []).append((start, stop))

    for i, gaps in enumerate(batches.values(), 1):
        positions = np.concatenate([np.arange(start, stop) for start, stop in gaps])
        for var in variables:
            arr = store[var]
            filled = []
            for start, stop in gaps:
                before, after = arr[start - 1].astype(np.float64), arr[stop].astype(np.float64)
                t0, t1 = times[start - 1], times[stop]
                w = ((times[start:stop] - t0) / (t1 - t0))[:, None, None]
                # NaN on either side (e.g. land) stays NaN
                filled.append(((1 - w) * before + w * after).astype(arr.dtype))
            arr[positions, :, :] = np.concatenate(filled)

        store["interp_mask"][positions] = True
        store["missing_mask"][positions] = False
        logger.info("[%s] batch %d/%d: filled %d steps", zarr_path.name, i, len(batches),
                    len(positions))

    return summary


def fill_gaps_output(cfg):
    ''' fill gaps in `<output_path>/<dataset>.zarr` for the selected dataset/domain '''
    if cfg.dataset.get("static", False):
        logger.info("[%s] static dataset (.npz), nothing to fill", cfg.dataset.name)
        return
    if cfg.get("gap_len") is None:
        raise ValueError("mode=fill_gaps needs gap_len (e.g. gap_len=6)")

    name = cfg.dataset.name.removeprefix(f"{cfg.dataset.source}_")
    fill_time_gaps(Path(cfg.output_path) / f"{name}.zarr", gap_len=cfg.gap_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill short time gaps in a Zarr store")
    parser.add_argument("zarr_path")
    parser.add_argument("--gap-len", type=int, required=True,
                        help="longest run of consecutive missing timesteps to fill")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    fill_time_gaps(args.zarr_path, args.gap_len, args.dry_run)
