'''
Input/Output functions for read and save data
'''

import os

import numpy as np
import xarray as xr
import dask.array as da
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







class ZarrTimeWriter:
    '''
    Save dataset in Zarr dataset format

    target_grid : tuple(lat, lon, y, x) -- squared
    '''

    def __init__(
        self,
        zarr_path,
        time_vector: np.datetime64,
        variable_names,
        target_grid,
        time_chunk=24,
        clevel=3,
        dtype=np.float32,
    ):
        self.zarr_path = zarr_path
        self.time_vector = time_vector
        self.variable_names = list(variable_names)
        self.grid = target_grid
        self.time_chunk = time_chunk
        self.clevel = clevel
        self.dtype = dtype

        if not os.path.exists(self.zarr_path):
            self._initialize()

        self.ds = xr.open_zarr(self.zarr_path)

    def _initialize(self):
        ''' init Zarr dataset '''

        nt = len(self.time_vector)
        nc = len(self.variable_names)
        h, w = self.grid[0].shape

        coords = {
            "variable": np.asarray(self.variable_names, dtype=str),
            "time": self.time_vector,
            "y": self.grid[2],
            "x": self.grid[3],
            "lat": (("y", "x"), self.grid[0]),
            "lon": (("y", "x"), self.grid[1]),
        }

        ds = xr.Dataset(coords=coords)

        # Create lazy empty arrays. (variable, time, y, x)
        data = da.full(
            (nc, nt, h, w),
            np.nan,
            chunks=(nc, self.time_chunk, h, w),
            dtype=self.dtype,
        )

        ds["data"] = (
            ("variable", "time", "y", "x"),
            data,
        )

        # True  = timestamp has not been written
        # False = timestamp has been written
        ds["nan_mask"] = (
            "time",
            da.ones(
                nt,
                chunks=(self.time_chunk,),
                dtype=bool,
            ),
        )

        compressor = BloscCodec(
            cname="zstd",
            clevel=self.clevel,
            shuffle="bitshuffle",
        )

        encoding = {
            "data": {
                "chunks": (
                    nc,
                    self.time_chunk,
                    h,
                    w,
                ),
                "compressors": [compressor],
            },
            "nan_mask": {
                "chunks": (self.time_chunk,),
            },
        }

        # Create only the Zarr structure/metadata.
        ds.to_zarr(
            self.zarr_path,
            mode="w",
            encoding=encoding,
            compute=False,
        )

    def write(self, data_dict: dict, time: np.ndarray):
        """Write data into the Zarr dataset."""

        # Check variables
        missing = set(self.variable_names) - set(data_dict)
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        # Find positions in global time_vector
        positions = np.searchsorted(self.time_vector, time)

        # stack vars to (variable, T, H, W):
        data = np.stack(list(data_dict.values()), axis=0)

        # write:
        self.ds['data'][:, positions, :, :] = data

        # False = data exists
        self.ds["nan_mask"][positions] = False

    def close(self):
        '''close dataset'''
        self.ds.close()
