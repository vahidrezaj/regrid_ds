'''Tests for read_tif'''

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from readers import read_tif


def _write_tif(path, data, crs, transform, nodata):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype),
        crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_read_tif(tmp_path):
    height, width = 4, 5
    nodata = -9999.0
    data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    data[0, 0] = nodata

    # projected CRS (UTM zone 32N, covers Denmark/North Sea), 1 km pixels
    crs = CRS.from_epsg(32632)
    transform = from_origin(500000, 6100000, 1000, 1000)
    tif_path = tmp_path / "bathy.tif"
    _write_tif(tif_path, data, crs, transform, nodata)

    ds_list = read_tif([tif_path], variable_names=["level"])

    assert len(ds_list) == 1
    ds = ds_list[0]
    assert ds["level"].shape == (height, width)
    assert ds["lat"].shape == (height, width)
    assert ds["lon"].shape == (height, width)

    # nodata pixel became NaN; a valid pixel keeps its value
    assert np.isnan(ds["level"].values[0, 0])
    assert ds["level"].values[1, 1] == data[1, 1]

    # reprojected to EPSG:4326: UTM zone 32N sits roughly within [0, 15]E, [40, 70]N
    assert np.all((ds["lon"].values > 0) & (ds["lon"].values < 15))
    assert np.all((ds["lat"].values > 40) & (ds["lat"].values < 70))


def test_read_tif_rejects_band_count_mismatch(tmp_path):
    crs = CRS.from_epsg(32632)
    transform = from_origin(500000, 6100000, 1000, 1000)
    tif_path = tmp_path / "bathy.tif"
    _write_tif(tif_path, np.zeros((3, 3), dtype=np.float32), crs, transform, -9999.0)

    with pytest.raises(ValueError):
        read_tif([tif_path], variable_names=["level", "extra"])


def test_read_tif_missing_crs_uses_override(tmp_path):
    # some raw exports (e.g. the real DMI bathymetry source) drop the embedded
    # CRS even though pixel coordinates are already plain WGS84 lon/lat
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    transform = from_origin(-5.0, 66.0, 0.5, 0.5)
    tif_path = tmp_path / "bathy_no_crs.tif"
    _write_tif(tif_path, data, None, transform, None)

    with pytest.raises(ValueError):
        read_tif([tif_path], variable_names=["level"])

    ds_list = read_tif([tif_path], variable_names=["level"], crs="EPSG:4326")
    ds = ds_list[0]
    # EPSG:4326 -> EPSG:4326 reprojection is the identity transform: pixel-center
    # coordinates pass through unchanged (origin -5.0/66.0, 0.5-degree pixels)
    assert np.allclose(ds["lon"].values[0, :], [-4.75, -4.25, -3.75, -3.25])
    assert np.allclose(ds["lat"].values[:, 0], [65.75, 65.25, 64.75])
    assert np.array_equal(ds["level"].values, data)
