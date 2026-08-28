'''
HBM Preprocessing class
'''

from pathlib import Path
from datetime import timedelta
from time import monotonic
import json
import logging
import os
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from hbm_prep.grid_interp import create_local_metric_grid, regridding_fn
from hbm_prep.io_functions import ZarrDataWriter, save_static_npz

logger = logging.getLogger(__name__)


def _to_plain(value):
    ''' resolve an OmegaConf node to a plain python object; pass through non-config values '''
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _fmt_duration(seconds):
    ''' format a duration in seconds as H:MM:SS '''
    return str(timedelta(seconds=int(seconds)))


class HBMPreProcessing:
    '''
    Read NetCDF source files, regrid them onto a common target grid, and write the
    result to a Zarr store, one source file (or one row of paired regional files) at a
    time. Progress is checkpointed to disk after every file so a crashed or interrupted
    run can be resumed by re-instantiating with the same config instead of starting over.

    Parameters
    ----------
    cfg : DictConfig
        Merged Hydra config with (at least) the following keys:

        - `dataset.folder` : source files subfolder, joined with `base_path`.
        - `dataset.variable_names` : source variable names to read/regrid.
        - `dataset.variable_attrs` : optional per-variable rename/attrs override,
          forwarded to `ZarrDataWriter`.
        - `dataset.interp_method` / `dataset.extrap_method` : passed to `regridding_fn`.
        - `dataset.pair_vars_list` : optional list of `[u, v]` variable-name pairs that
          need rotation-aware regridding, passed to `regridding_fn`.
        - `dataset.reader_fn` : a `_partial_: true` target instantiated into `loader_fn`,
          called with a list of file paths (one per region) and returning a matching
          list of opened datasets.
        - `dataset.name` : used to namespace the checkpoint/output file, and to look up
          this dataset's entry in `preprocessing.file_prefix`.
        - `dataset.file_ext` : source file extension to glob for, default `".nc"`.
        - `dataset.static` : bool, default `False`. `True` marks a time-invariant
          source (e.g. a bathymetry raster): no `time_vector`/`ZarrDataWriter`/
          per-file checkpoint, just one regrid, saved to a `.npz` file. See
          `_process_static`.
        - `preprocessing.file_prefix` : dict of `{dataset_name: [prefix, ...] or null}`.
          Each prefix becomes one region's file queue
          (`data_path.glob(prefix + "*" + file_ext)`, sorted); a missing entry or
          `null` falls back to a single unprefixed queue. Queues are advanced in
          lockstep, so all regions must have equal file counts.
        - `preprocessing.domain_size` / `grid_size` / `lat_0` / `lon_0` : passed to
          `create_local_metric_grid` to build the target grid.
        - `preprocessing.from_to` / `ts` : start/end timestamps and step (hours) defining
          `time_vector`, the full output time axis. Unused when `dataset.static` is `True`.
        - `preprocessing.time_chunk` / `clevel` : forwarded to `ZarrDataWriter`.
          Unused when `dataset.static` is `True`.
        - `output_path` : base directory for the checkpoint file and Zarr store /
          `.npz` file.
        - `verbose` : bool, default `False`. `True` raises the module logger to
          `DEBUG`, adding skip reasons and per-step (read/regrid/write) timings on
          top of the always-on per-file `INFO` progress line.

    base_path : str or Path, optional
        Prepended to `cfg.dataset.folder`.

    Call
    ----
    For time-series datasets, `__call__()` runs the full loop: for each row of files
    still queued (in priority order, per `regridding_fn`), read and regrid the ones
    falling inside `time_vector`, write them to the Zarr store, then update/delete the
    checkpoint. Safe to call again on an already-completed instance (no-op, since its
    checkpoint file is gone).

    For static datasets (`dataset.static: true`), `__call__()` instead delegates to
    `_process_static`: reads and regrids the (single) row of files once and saves the
    result to a `.npz` file, skipping entirely if that file already exists.
    '''
    def __init__(self, cfg, base_path=""):
        self.dataset_name = cfg.dataset.name
        self.verbose = bool(cfg.get("verbose", False))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        self.data_path = Path(base_path) / Path(cfg.dataset.folder)
        self.out_path = Path(cfg.output_path)

        self.variable_names = list(cfg.dataset.variable_names)
        self.variable_attrs = _to_plain(cfg.dataset.get("variable_attrs", None))
        self.static = bool(cfg.dataset.get("static", False))
        file_ext = cfg.dataset.get("file_ext", ".nc")

        # init checkpoint with available files (time-series datasets only):
        self.cp_path = self.out_path / f"checkpoint_{self.dataset_name}.tmp"
        self.avail_files = []
        if not self.static and self.cp_path.exists():
            with self.cp_path.open("r", encoding="utf-8") as f:
                self.avail_files = [[Path(p) for p in row] for row in json.load(f)]

        if len(self.avail_files) == 0:
            prefixes = _to_plain(cfg.preprocessing.file_prefix.get(self.dataset_name)) or [""]
            self.avail_files = [sorted(self.data_path.glob(pref + "*" + file_ext))
                                for pref in prefixes]

        # check length mismatch
        lengths = {len(row) for row in self.avail_files}
        if len(lengths) > 1:
            raise ValueError(
                f"file count mismatch across dataset regions: "
                f"{[len(row) for row in self.avail_files]}"
            )
        self.total_files = (len(self.avail_files[0])
                            if self.avail_files and self.avail_files[0] else 0)

        # generate target grid:
        self.target_grid = create_local_metric_grid(
            domain_size_km= cfg.preprocessing.domain_size,
            grid_size= cfg.preprocessing.grid_size,
            lat_0= cfg.preprocessing.lat_0,
            lon_0= cfg.preprocessing.lon_0,
            proj_type= 'aeqd',
        )

        # file reader function:
        self.loader_fn = instantiate(cfg.dataset.reader_fn)

        self.interp_method = _to_plain(cfg.dataset.interp_method)
        self.extrap_method = _to_plain(cfg.dataset.extrap_method)
        self.pair_vars_list = _to_plain(cfg.dataset.get("pair_vars_list", []))

        if self.static:
            # no time axis: one-shot regrid, saved as .npz (see _process_static)
            self.npz_path = self.out_path / f"{self.dataset_name}.npz"
            return

        # time vector:
        self.time_vector = np.arange(
            np.datetime64(cfg.preprocessing.from_to[0]),
            np.datetime64(cfg.preprocessing.from_to[1]),
            np.timedelta64(cfg.preprocessing.ts, "h"),
        )

        # dataset writer:
        self.writer = ZarrDataWriter(
            zarr_path= self.out_path / f"checkpoint_{self.dataset_name}.zarr",
            time_vector= self.time_vector,
            variable_names= self.variable_names,
            target_grid= self.target_grid,
            variable_attrs= self.variable_attrs,
            time_chunk=cfg.preprocessing.time_chunk,
            clevel=cfg.preprocessing.clevel,
        )

    def __call__(self):
        ''' read -> regrid -> write ; static datasets run once, others loop per file '''
        if self.static:
            self._process_static()
            return

        # loop over files:
        start_time = monotonic()
        processed = 0
        while self.avail_files and self.avail_files[0]:
            iter_start = monotonic()
            files = [row[0] for row in self.avail_files]

            # read files:
            ds_list = self.loader_fn(files)
            read_time = monotonic()

            # trim time out of the time_vector range:
            time = ds_list[0].time.values
            time = np.asarray(time, dtype=self.time_vector.dtype)
            time_mask = (time >= self.time_vector[0]) & \
                        (time <= self.time_vector[-1])
            time = time[time_mask]
            if time.size == 0:
                # this file doesn't have data in the range of time_vector
                logger.debug("[%s] skipping %s (outside time range)", self.dataset_name, files)
                self.avail_files = [row[1:] for row in self.avail_files]
                self._update_cp()
                continue

            # regridding into the target grid:
            ds = regridding_fn(
                ds_list,
                self.target_grid,
                self.variable_names,
                time_mask,
                self.interp_method,
                self.extrap_method,
                self.pair_vars_list,
            )
            regrid_time = monotonic()

            # write Zarr dataset:
            self.writer.write(ds)
            write_time = monotonic()

            # update avail_files and chekpoint
            self.avail_files = [row[1:] for row in self.avail_files]
            self._update_cp()

            processed += 1
            remaining = len(self.avail_files[0]) if self.avail_files else 0
            elapsed = write_time - start_time
            avg = elapsed / processed
            eta = avg * remaining
            logger.info(
                "[%s] %d/%d done, %d left | %s | %.1fs (avg %.1fs/file, elapsed %s, ETA %s)",
                self.dataset_name, processed, self.total_files, remaining,
                ", ".join(f.name for f in files), write_time - iter_start,
                avg, _fmt_duration(elapsed), _fmt_duration(eta),
            )
            logger.debug(
                "[%s] step timings: read %.1fs, regrid %.1fs, write %.1fs",
                self.dataset_name, read_time - iter_start,
                regrid_time - read_time, write_time - regrid_time,
            )

        # everything completed successfully, so let's delete cp file :)
        self.cp_path.unlink(missing_ok=True)
        logger.info(
            "[%s] completed: %d files processed in %s",
            self.dataset_name, processed, _fmt_duration(monotonic() - start_time),
        )

    def _update_cp(self):
        ''' update checkpoint file '''
        tmp_path = self.cp_path.with_name(self.cp_path.name + ".part")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump([[str(p) for p in row] for row in self.avail_files], f)
        os.replace(tmp_path, self.cp_path)

    def _process_static(self):
        ''' one-shot regrid + save for time-invariant datasets (e.g. bathymetry) '''
        if self.npz_path.exists():
            logger.info("[%s] %s already exists, skipping", self.dataset_name, self.npz_path)
            return

        files = [row[0] for row in self.avail_files]
        ds_list = self.loader_fn(files)

        ds = regridding_fn(
            ds_list,
            self.target_grid,
            self.variable_names,
            None,
            self.interp_method,
            self.extrap_method,
            self.pair_vars_list,
        )

        arrays = {
            self._store_name(var): np.asarray(ds[var].values)
            for var in self.variable_names
        }
        save_static_npz(self.npz_path, arrays, self.target_grid)

    def _store_name(self, var):
        ''' source variable name -> name to use in the output, per variable_attrs '''
        if not self.variable_attrs:
            return var
        return self.variable_attrs.get(var, {}).get("name", var)
