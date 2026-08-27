'''
Input/Output functions for read and save data
'''

import os

import numpy as np
import xarray as xr
import dask.array as da
import zarr
from zarr.codecs import BloscCodec


def read_nc(files:list) -> list:
    '''
    Read nc files
    
    Returns : list of loaded ds 
    '''
    # load ds
    ds_list = []
    for file in files:
        ds_list.append(xr.open_dataset(file))

    # check time files:
    if len(ds_list) > 1:
        assert all(ds.time.equals(ds_list[0].time) for ds in ds_list[1:]), \
        f"Time vectors of the datasets are not identical. Files: {files}"

    return ds_list



# def read_tif_file():
#     pass







class ZarrDataWriter:
    '''
    Pre-allocate a Zarr store sized for the full run and fill it in incrementally,
    one array per variable plus a `nan_mask` marking which timestamps have been written.

    Parameters
    ----------
    zarr_path : str or Path
        Output .zarr store location. Created if missing; if it already exists, its configuration
        is validated against the parameters below instead of being re-initialized.
    time_vector : np.ndarray of np.datetime64
        Full time axis for the run (e.g. the entire multi-year period), sorted ascending.
        `write()` maps each chunk's timestamps onto this axis by exact match.
    variable_names : list of str
        Variables expected in each `ds` passed to `write()`, keyed by their *source* name.
    target_grid : dict
        Target grid, as returned by `create_local_metric_grid`: 'lat', 'lon' (2-D, shape (y, x)),
        'y', 'x' (1-D projected coords, metres), and 'crs' (CF grid-mapping attrs dict from
        `CRS.to_cf()`).
    variable_attrs : dict, optional
        `{source_name: {"name": store_name, "units": ..., ...}}`. When a source variable has an 
        entry here, its data is stored under `store_name` instead of `source_name`, and these 
        attrs are used verbatim instead of `ds[var].attrs`
    time_chunk : int, default 24
        Number of timesteps per Zarr chunk along the time axis.
    clevel : int, default 3
        Blosc/zstd compression level.
    dtype : np.floating, default np.float32
        Storage dtype for data variables. Must be floating-point.

    Notes
    -----
    Gaps in the source data (timestamps never passed to `write()`) are
    left as NaN with `nan_mask=True`, rather than written explicitly.
    '''

    # attrs we manage ourselves; never overwritten by source variable attrs
    _RESERVED_ATTRS = {"grid_mapping", "_FillValue", "missing_value"}

    def __init__(
        self,
        zarr_path,
        time_vector: np.ndarray,
        variable_names,
        target_grid,
        variable_attrs=None,
        time_chunk=24,
        clevel=3,
        dtype=np.float32,
    ):
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(
                "dtype must be floating-point: unwritten cells are read back "
                "as fill_value=NaN"
            )

        self.zarr_path = zarr_path
        self.time_vector = np.asarray(time_vector)
        self.variable_names = list(variable_names)
        self.grid = target_grid
        self.variable_attrs = variable_attrs or {}
        self.time_chunk = time_chunk
        self.clevel = clevel
        self.dtype = dtype

        # variables whose descriptive attrs have already been captured
        self._attrs_written = set()

        # source name -> name actually used in the store:
        self._store_name = {
            var: self.variable_attrs.get(var, {}).get("name", var)
            for var in self.variable_names
        }
        store_names = list(self._store_name.values())
        if len(set(store_names)) != len(store_names):
            raise ValueError(f"duplicate target variable names in variable_attrs: {store_names}")

        if os.path.exists(self.zarr_path):
            self._validate_existing()
        else:
            self._initialize()

        self.store = zarr.open_group(self.zarr_path, mode="a")

    def _validate_existing(self):
        ''' check an existing store matches this writer's configuration '''
        existing = xr.open_zarr(self.zarr_path, consolidated=True)
        try:
            if not np.array_equal(existing["time"].values, self.time_vector):
                raise ValueError(
                    f"time_vector mismatch with existing store at {self.zarr_path}"
                )

            missing = set(self._store_name.values()) - set(existing.data_vars)
            if missing:
                raise ValueError(
                    f"existing store at {self.zarr_path} is missing variables: {missing}"
                )

            h, w = self.grid['lat'].shape
            if existing.sizes["y"] != h or existing.sizes["x"] != w:
                raise ValueError(
                    f"grid shape mismatch with existing store at {self.zarr_path}"
                )

            existing_crs = existing["spatial_ref"].attrs
            new_crs = self.grid['crs']
            for key in ("latitude_of_projection_origin", "longitude_of_projection_origin"):
                if not np.isclose(existing_crs.get(key, np.nan), new_crs.get(key, np.nan)):
                    raise ValueError(
                        f"CRS origin mismatch with existing store at {self.zarr_path}: "
                        f"{key}={new_crs.get(key)!r} vs stored {existing_crs.get(key)!r}"
                    )
        finally:
            existing.close()

    def _initialize(self):
        ''' init Zarr dataset '''

        nt = len(self.time_vector)
        h, w = self.grid['lat'].shape

        coords = {
            "time": self.time_vector,
            "y": ("y", self.grid['y'], {
                "units": "m",
                "standard_name": "projection_y_coordinate",
            }),
            "x": ("x", self.grid['x'], {
                "units": "m",
                "standard_name": "projection_x_coordinate",
            }),
            "lat": (("y", "x"), self.grid['lat'], {
                "units": "degrees_north",
                "standard_name": "latitude",
            }),
            "lon": (("y", "x"), self.grid['lon'], {
                "units": "degrees_east",
                "standard_name": "longitude",
            }),
            # CF grid-mapping variable: dummy scalar value, real content is attrs
            "spatial_ref": ((), 0, self.grid['crs']),
        }

        ds = xr.Dataset(coords=coords, attrs={"Conventions": "CF-1.8"})

        compressor = BloscCodec(
            cname="zstd",
            clevel=self.clevel,
            shuffle="bitshuffle",
        )

        # lazy initailization:
        encoding = {}
        for var in self.variable_names:
            store_name = self._store_name[var]
            ds[store_name] = (
                ("time", "y", "x"),
                da.empty((nt, h, w), chunks=(self.time_chunk, h, w), dtype=self.dtype),
                {"grid_mapping": "spatial_ref"},
            )
            encoding[store_name] = {
                "chunks": (self.time_chunk, h, w),
                "compressors": [compressor],
                "fill_value": np.nan,
            }

        # True  = timestamp has not been written
        # False = timestamp has been written
        ds["nan_mask"] = (
            "time",
            da.empty(nt, chunks=(self.time_chunk,), dtype=bool),
        )
        encoding["nan_mask"] = {
            "chunks": (self.time_chunk,),
            "fill_value": True,
        }

        # Create only the Zarr structure/metadata:
        ds.to_zarr(
            self.zarr_path,
            mode="w",
            encoding=encoding,
            compute=False,
        )

        zarr.consolidate_metadata(self.zarr_path)

    def write(self, ds: xr.Dataset):
        """Write a chunk of already-regridded data into the Zarr store.

        `ds` must have a `time` coordinate and a data variable for each of
        `self.variable_names`. Timestamps not present in `ds` are simply
        left untouched. gaps in the source data are represented by
        omission (they stay at fill_value=NaN / nan_mask=True), not by
        writing NaN explicitly.
        """

        missing = set(self.variable_names) - set(ds.data_vars)
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        time = ds["time"].values

        # Find positions in global time_vector:
        positions = np.searchsorted(self.time_vector, time)

        # check: searchsorted can return an out-of-bounds index
        in_bounds = positions < len(self.time_vector)
        matched = np.zeros(len(time), dtype=bool)
        matched[in_bounds] = self.time_vector[positions[in_bounds]] == time[in_bounds]
        if not np.all(matched):
            raise ValueError(f"time values not found in time_vector: {time[~matched]}")

        attrs_changed = False
        for var in self.variable_names:
            store_name = self._store_name[var]

            self.store[store_name][positions, :, :] = ds[var].transpose("time", "y", "x").values

            # keep attrs (units, long_name, ...)
            if var not in self._attrs_written:
                # from variable_attrs, if None form the source variable
                override = self.variable_attrs.get(var)
                source_attrs = override if override is not None else ds[var].attrs
                var_attrs = {
                    k: v for k, v in source_attrs.items()
                    if k not in self._RESERVED_ATTRS and k != "name"
                }
                if var_attrs:
                    self.store[store_name].attrs.update(var_attrs)
                    attrs_changed = True
                self._attrs_written.add(var)

        # False = data exists
        self.store["nan_mask"][positions] = False

        if attrs_changed:
            # consolidated metadata to update attrs from its original file
            zarr.consolidate_metadata(self.zarr_path)

    def close(self):
        '''No-op in the current impelmentation:
        writes go straight through the zarr array API and, so there is nothing left to flush.'''
