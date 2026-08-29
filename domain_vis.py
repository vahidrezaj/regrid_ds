'''
regrid one sample of whichever `dataset=` to visually confirm domain/regrdding
'''

import os
from pathlib import Path
import random
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from hbm_prep.grid_interp import create_local_metric_grid, regridding_fn


load_dotenv()
base_path = Path(os.getenv("LOCAL_DIR"))

# maps grid_interp's proj4 `proj_type` codes to the matching cartopy projection
_PROJECTIONS = {
    "aeqd": ccrs.AzimuthalEquidistant,
    "laea": ccrs.LambertAzimuthalEqualArea,
}


def _to_plain(value):
    ''' resolve an OmegaConf node to a plain python object; pass through non-config values '''
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _apply_variable_attrs(ds, variable_attrs):
    ''' overlay `dataset.variable_attrs` (units/long_name/...) onto the regridded ds, mirroring
    what `ZarrDataWriter.write` does at write time, so plot labels match the real output '''
    if not variable_attrs:
        return ds
    for var, override in variable_attrs.items():
        if var in ds.data_vars:
            ds[var].attrs.update({k: v for k, v in override.items() if k != "name"})
    return ds


def _raw_slice(ds, var, mask):
    ''' mirror regridding_fn's per-source time/depth selection, before regridding '''
    da = ds[var]
    if mask is None:
        return da
    da = da[mask, 0] if da.ndim > 3 else da[mask]
    return da.isel(time=0)


def _perimeter(lat, lon):
    ''' trace the outer edge of a (curvilinear or regular) lat/lon coordinate array '''
    if lat.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    return (
        np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]]),
        np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]]),
    )


def _plot_variable(cfg, var, ds, ds_list, mask, target_grid, proj_type):
    ''' one figure for `var`: native-projection source regions next to the regridded result '''
    da = ds[var] if mask is None else ds[var].isel(time=0)
    vmin, vmax = np.nanmin(da.values), np.nanmax(da.values)

    raw_slices = [_raw_slice(source, var, mask) for source in ds_list]
    target_perimeter = _perimeter(target_grid["lat_b"], target_grid["lon_b"])
    source_perimeters = [_perimeter(raw.lat.values, raw.lon.values) for raw in raw_slices]

    regrid_proj = _PROJECTIONS[proj_type](
        central_longitude=cfg.preprocessing.lon_0, central_latitude=cfg.preprocessing.lat_0,
    )
    source_proj = _PROJECTIONS[proj_type](
        central_longitude=cfg.preprocessing.lon_0, central_latitude=cfg.preprocessing.lat_0,
    )

    fig = plt.figure(constrained_layout=True, figsize=(12, 5))
    ax_source = fig.add_subplot(1, 2, 1, projection=source_proj)
    ax_regrid = fig.add_subplot(1, 2, 2, projection=regrid_proj)

    for raw, (lon_p, lat_p) in zip(raw_slices, source_perimeters):
        mesh = ax_source.pcolormesh(
            raw.lon.values, raw.lat.values, raw.values,
            shading="auto", transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
        )
        ax_source.plot(lon_p, lat_p, color="gray", linewidth=1, transform=ccrs.PlateCarree())
    ax_source.coastlines(resolution="10m")
    ax_source.set_title("source regions")

    mesh = ax_regrid.pcolormesh(
        target_grid["lon_b"], target_grid["lat_b"], da.values,
        shading="flat", transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
    )
    ax_regrid.plot(*target_perimeter, color="gray", linewidth=1, transform=ccrs.PlateCarree())
    ax_regrid.coastlines(resolution="10m")
    ax_regrid.set_title(f"regridded  [{proj_type.upper()}]")

    # zoom out a bit so the regridded panel's own boundary box isn't sitting flush on the frame edge
    pad = 0.03 * (target_grid["x"].max() - target_grid["x"].min())
    ax_regrid.set_extent([
        target_grid["x"].min() - pad, target_grid["x"].max() + pad,
        target_grid["y"].min() - pad, target_grid["y"].max() + pad,
    ], crs=regrid_proj)

    # vmin/vmax are shared, so one colorbar (from the regridded mesh) covers both axes
    units = da.attrs.get("units", "")
    label = f"{cfg.dataset.name}: {var}" + (f" [{units}]" if units else "")
    fig.colorbar(mesh, ax=[ax_source, ax_regrid], label=label)

    return fig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    '''
    main
    '''
    data_path = base_path / Path(cfg.dataset.folder)
    file_ext = cfg.dataset.get("file_ext", ".nc")
    prefixes = _to_plain(cfg.preprocessing.file_prefix.get(cfg.dataset.name)) or [""]
    files = [sorted(data_path.glob(pref + "*" + file_ext))[0] for pref in prefixes]

    proj_type = "aeqd"
    target_grid = create_local_metric_grid(
        domain_size_km=cfg.preprocessing.domain_size,
        grid_size=cfg.preprocessing.grid_size,
        lat_0=cfg.preprocessing.lat_0,
        lon_0=cfg.preprocessing.lon_0,
        proj_type=proj_type,
    )

    loader_fn = instantiate(cfg.dataset.reader_fn)
    ds_list = loader_fn(files)

    variable_names = _to_plain(cfg.dataset.variable_names)
    pair_vars_list = _to_plain(cfg.dataset.get("pair_vars_list", []))
    static = bool(cfg.dataset.get("static", False))
    use_mask = bool(cfg.dataset.get("use_mask", True))

    if static:
        mask = None
        ds = regridding_fn(ds_list, target_grid, variable_names, None,
                            cfg.dataset.interp_method, cfg.dataset.extrap_method, pair_vars_list,
                            use_mask)
    else:
        mask = np.zeros(ds_list[0].sizes["time"], dtype=bool)
        i = random.randrange(len(mask)) # random timestamp
        mask[i] = True
        ds = regridding_fn(ds_list, target_grid, variable_names, mask,
                            cfg.dataset.interp_method, cfg.dataset.extrap_method, pair_vars_list,
                            use_mask)

    ds = _apply_variable_attrs(ds, _to_plain(cfg.dataset.get("variable_attrs", None)))

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    for var in variable_names:
        fig = _plot_variable(cfg, var, ds, ds_list, mask, target_grid, proj_type)
        fig_path = fig_dir / f"{cfg.preprocessing.name}_{cfg.dataset.name}_{var}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"saved {fig_path}")


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
