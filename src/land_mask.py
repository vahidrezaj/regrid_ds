'''
Shared land mask for a domain, built once (`mode=land_mask`) and used by every dataset
with `land_mask: true`.

- Inside the ocean model's coverage: the model's land (cells < 50% ocean).
- Outside it: Natural Earth land. Sea there is filled later (see `spatial_fill.py`),
  but only if it borders the model's sea; other sea is marked land (no data).
'''

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe
from hydra.utils import instantiate
from scipy.ndimage import binary_dilation, label

from grid_interp import create_local_metric_grid
from writers import save_static_npz

logger = logging.getLogger(__name__)

LAND_MASK_FILE = "land_mask.npz"


def _sample_files(cfg):
    ''' first file of each domain.file_match queue '''
    data_path = Path(cfg.dataset.folder)
    file_ext = cfg.dataset.get("file_ext", ".nc")
    files = []
    for tok in cfg.domain.file_match.get(cfg.dataset.name) or [""]:
        pattern = f"*{tok}*{file_ext}" if tok else f"*{file_ext}"
        matches = sorted(data_path.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no {pattern} file in {data_path}")
        files.append(matches[0])
    return files


def _region_ocean(ds, variable_name, target_grid):
    ''' (ocean, covered) on the target grid for one source region '''
    if "source_mask" in ds:
        src_ocean = ds["source_mask"]
    else:
        src_ocean = ds[variable_name].notnull()
    if "time" in src_ocean.dims:
        src_ocean = src_ocean.isel(time=0)

    ds_source = xr.Dataset(coords={"lat": ds.lat, "lon": ds.lon})
    ds_target = xr.Dataset(coords={
        "lat": (("y", "x"), target_grid["lat"]),
        "lon": (("y", "x"), target_grid["lon"]),
    })
    regridder = xe.Regridder(ds_source, ds_target, "bilinear", unmapped_to_nan=True)

    fraction = np.asarray(regridder(src_ocean.astype(float)))
    covered = np.isfinite(np.asarray(regridder(xr.ones_like(src_ocean, dtype=float))))
    return np.nan_to_num(fraction) > 0.5, covered


def reference_land(lat, lon, shapefile=None):
    ''' True where (lat, lon) is on land, from Natural Earth 10 m (or `shapefile`) '''
    import cartopy.io.shapereader as shpreader  # pylint: disable=import-outside-toplevel
    import shapely  # pylint: disable=import-outside-toplevel

    path = shapefile or shpreader.natural_earth(
        resolution="10m", category="physical", name="land",
    )
    pad = 1.0
    box = shapely.box(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
    land = shapely.union_all([
        g.intersection(box) for g in shpreader.Reader(path).geometries() if g.intersects(box)
    ])
    shapely.prepare(land)
    return shapely.contains_xy(land, lon, lat)


def _parts_touching(mask, other):
    ''' connected parts of `mask` that touch `other` '''
    parts, _ = label(mask)
    ids = np.unique(parts[binary_dilation(other) & mask])
    return np.isin(parts, ids[ids > 0])


def build_land_mask(cfg):
    '''
    `mode=land_mask`
    Build `<output_path>/land_mask.npz` from an ocean dataset (e.g. `dataset=hbm_ocean`).
    Skipped if the file exists; delete it to rebuild.

    `+land_reference=<shapefile>` replaces Natural Earth.
    '''
    npz_path = Path(cfg.output_path) / LAND_MASK_FILE
    if npz_path.exists():
        logger.info("%s already exists, skipping (delete it to rebuild)", npz_path)
        return

    target_grid = create_local_metric_grid(
        domain_size_km=cfg.domain.domain_size,
        grid_size=cfg.domain.grid_size,
        lat_0=cfg.domain.lat_0,
        lon_0=cfg.domain.lon_0,
        proj_type="aeqd",
        alpha_deg=cfg.domain.get("alpha_deg", 0.0),
    )
    files = _sample_files(cfg)
    ds_list = instantiate(cfg.dataset.reader_fn)(files)
    variable_name = cfg.dataset.variable_names[0]

    ocean = np.zeros(target_grid["lat"].shape, dtype=bool)
    covered = np.zeros_like(ocean)
    for ds in ds_list:
        region_ocean, region_covered = _region_ocean(ds, variable_name, target_grid)
        ocean |= region_ocean
        covered |= region_covered

    ref_land = reference_land(target_grid["lat"], target_grid["lon"], cfg.get("land_reference"))

    # land that Natural Earth calls sea and borders the open boundary is treated as sea;
    # all other land (fjords, islets) stays as land
    open_sea = _parts_touching(covered & ~ocean & ~ref_land, ~covered & ~ref_land)
    land = np.where(covered, ~ocean & ~open_sea, ref_land)

    # sea not connected to the model's sea would be filled across land, so mark it land (no data)
    to_fill = ~land & ~ocean
    cut_off = to_fill & ~_parts_touching(to_fill, ocean)
    land |= cut_off

    save_static_npz(npz_path, {"land_mask": land, "covered": covered}, target_grid)
    logger.info(
        "saved %s: land %.1f%%, sea to fill %.1f%%, sea cut off %.1f%%",
        npz_path, 100 * land.mean(), 100 * (~land & ~ocean).mean(), 100 * cut_off.mean(),
    )
    _plot(cfg, target_grid, land, covered, ocean, cut_off)


def load_land_mask(out_path, target_grid):
    ''' (y, x) bool land mask from `<out_path>/land_mask.npz`, checked against `target_grid` '''
    npz_path = Path(out_path) / LAND_MASK_FILE
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found: build it first, e.g. "
            "`python run.py mode=land_mask dataset=hbm_ocean domain=<domain>`"
        )
    with np.load(npz_path, allow_pickle=True) as payload:
        lat, lon = payload["lat"], payload["lon"]
        same_grid = (
            np.shape(lat) == np.shape(target_grid["lat"])
            and np.allclose(lat, target_grid["lat"], atol=1e-6)
            and np.allclose(lon, target_grid["lon"], atol=1e-6)
        )
        if not same_grid:
            raise ValueError(f"{npz_path} was built for a different target grid, rebuild it")
        return np.asarray(payload["land_mask"], dtype=bool)


def _plot(cfg, target_grid, land, covered, ocean, cut_off):
    ''' quick-look figure: model land / reference land / cut-off sea / ocean / ocean to fill '''
    # pylint: disable=import-outside-toplevel
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = np.select(
        [cut_off, covered & land, ~covered & land, ocean], [0, 1, 2, 3], default=4,
    )
    labels = [
        "sea, cut off (no data)", "land (model)", "land (Natural Earth)",
        "ocean (model)", "ocean, to fill",
    ]
    cmap = matplotlib.colors.ListedColormap(
        ["#9aa5b1", "#8c6d46", "#c9b18f", "#3b7dd8", "#f2a541"],
    )

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=cfg.domain.lon_0, central_latitude=cfg.domain.lat_0,
    )
    fig = plt.figure(figsize=(7, 6.5), constrained_layout=True)
    ax = fig.add_subplot(projection=proj)
    mesh = ax.pcolormesh(
        target_grid["lon_b"], target_grid["lat_b"], classes, cmap=cmap, vmin=-0.5, vmax=4.5,
        shading="flat", transform=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.set_title(f"{cfg.domain.name}: land mask from {cfg.dataset.name}")
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(5), shrink=0.8)
    cbar.ax.set_yticklabels(labels)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / f"{cfg.domain.name}_{cfg.dataset.source}_land_mask.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info("saved %s", fig_path)
