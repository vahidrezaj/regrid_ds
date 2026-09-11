'''
Dataset-agnostic source-file readers: plain NetCDF (`read_nc`) and single-band-per-variable
GeoTIFF (`read_tif`). Source-specific reading (NEMO's curvilinear-grid unrotation, etc.) lives
in its own module (e.g. `nemo_reader.py`) instead.
'''

import numpy as np
import rioxarray
import xarray as xr
from pyproj import CRS, Transformer


def _decode_nonstandard_time(ds, file):
    '''
    Some source files use `units: "day as %Y%m%d.%f"` instead (e.g. 20131001.25 ==
    2013-10-01 06:00), which cause silent gap in downstream by skipping regridding
    and saving.
    No-op when `time` already decoded to datetime64 (the common case).
    '''
    time = ds["time"]
    if np.issubdtype(time.dtype, np.datetime64):
        return ds

    units = time.attrs.get("units", "")
    if units != "day as %Y%m%d.%f":
        raise ValueError(f"{file}: unrecognized/undecoded time units {units!r}")

    raw = np.atleast_1d(time.values).astype(np.float64)
    day_part = np.floor(raw + 1e-6).astype(np.int64)
    seconds = np.round((raw - day_part) * 86400).astype(np.int64)
    decoded = np.array([
        np.datetime64(f"{d // 10000:04d}-{(d // 100) % 100:02d}-{d % 100:02d}")
        + np.timedelta64(int(s), "s")
        for d, s in zip(day_part, seconds)
    ])
    return ds.assign_coords(time=("time", decoded))


def read_nc(files:list) -> list:
    '''
    Read nc files

    Returns : list of loaded ds
    '''
    # load ds
    ds_list = []
    for file in files:
        ds = xr.open_dataset(file)
        ds = _decode_nonstandard_time(ds, file)
        ds_list.append(ds)

    # check time files:
    if len(ds_list) > 1:
        assert all(ds.time.equals(ds_list[0].time) for ds in ds_list[1:]), \
        f"Time vectors of the datasets are not identical. Files: {files}"

    return ds_list



def read_tif(
        files: list, variable_names: list, crs=None, resolution_km: float | None = None,
) -> list:
    '''
    Read single-band-per-variable GeoTIFF rasters (e.g. a static bathymetry grid).

    Each file's band(s) become one data variable per entry in `variable_names`
    (band i -> variable_names[i]). Coordinates are built as 2-D "lat"/"lon"
    (dims "y", "x"), reprojected to EPSG:4326 from the raster's CRS -- matching
    the curvilinear lat/lon-as-2-D-coords shape already used for NEMO ocean
    sources, so downstream regridding needs no special-casing for tif sources.
    Nodata pixels are read back as NaN.

    crs : optional CRS (anything accepted by `pyproj.CRS.from_user_input`,
        e.g. "EPSG:4326"), used only when the raster itself has no CRS
        embedded.
    resolution_km : optional target pixel size in km. When given, the raster is
        block-averaged (before reprojecting) down to approximately this resolution.

    Returns : list of loaded ds
    '''
    def coarsen_to_resolution(raster, crs):
        ''' block-average `raster` (dims "y", "x") down to ~resolution_km per pixel, based
        on its own native pixel size (converted from degrees to km at the raster's mean
        latitude, if `crs` is geographic). Returns `raster` unchanged if it's already
        coarser than that. '''
        km_per_lat = 111.32
        res_x, res_y = raster.rio.resolution()
        if crs.is_geographic:
            mean_lat = float(raster.y.values.mean())
            km_per_deg_x = km_per_lat * np.cos(np.deg2rad(mean_lat))
            native_km_x, native_km_y = abs(res_x) * km_per_deg_x, abs(res_y) * km_per_lat
        else:
            # pixel size is already in the CRS's linear unit (metres)
            native_km_x, native_km_y = abs(res_x) / 1000, abs(res_y) / 1000

        stride_x = max(1, round(resolution_km / native_km_x))
        stride_y = max(1, round(resolution_km / native_km_y))
        if stride_x == 1 and stride_y == 1:
            return raster
        return raster.coarsen(y=stride_y, x=stride_x, boundary="trim").mean()

    ds_list = []
    for file in files:
        with rioxarray.open_rasterio(file, masked=True) as raster:
            n_bands = raster.sizes["band"]
            if n_bands != len(variable_names):
                raise ValueError(
                    f"{file}: raster has {n_bands} band(s), but {len(variable_names)} "
                    f"variable_names were given: {variable_names}"
                )

            src_crs = raster.rio.crs or crs
            if src_crs is None:
                raise ValueError(
                    f"{file}: raster has no embedded CRS and no `crs` override was given"
                )
            src_crs = CRS.from_user_input(src_crs)

            if resolution_km:
                raster = coarsen_to_resolution(raster, src_crs)

            # native pixel-center coordinates -> 2-D lat/lon in EPSG:4326
            x_mg, y_mg = np.meshgrid(raster.x.values, raster.y.values)
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_mg, y_mg)

            data = {var: raster.isel(band=i).values for i, var in enumerate(variable_names)}

        ds = xr.Dataset(
            {var: (("y", "x"), values) for var, values in data.items()},
            coords={
                "lat": (("y", "x"), lat),
                "lon": (("y", "x"), lon),
            },
        )
        ds_list.append(ds)

    return ds_list
