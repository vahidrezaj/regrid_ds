'''
Does rotating the target grid (alpha_deg) change the physical wind/current
field, or just which axes u/v are measured against? Two checks:

1. Numeric: rotate a synthetic wind vector into the grid's local basis and
   back, for alpha_deg=0 and the real configured value. A rotation is always
   exactly invertible, so this should recover the original vector almost
   perfectly. Also checks how much error the grid's existing rotation-only
   (no shear correction) approximation carries at this domain's size.
2. Visual: regrid one real ERA5 wind timestep through the actual pipeline
   twice -- alpha_deg=0 vs. the configured value -- un-rotate both back to
   true east/north, and plot them together. If alpha_deg preserves the
   field, both should trace the same wind pattern. (Wind rather than ocean
   current, because current is noisy at small scales -- two different
   sample points of the same current field can look different for reasons
   that have nothing to do with alpha_deg.)

Edit FILE_INDEX / TIME_INDEX below to look at a different file/timestep.
Run `python tests/alpha_vector_checker.py` from the repo root (same env as
domain_vis.py). Saves figures/alpha_vector_check.png.
'''

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")  # no display
# pylint: disable=wrong-import-position
import cartopy.crs as ccrs
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from pyproj import Proj
from scipy.interpolate import griddata

from grid_interp import RegridPipeline, _rotate_vectors, create_local_metric_grid

# create_local_metric_grid's own benchmark: "<~0.3 deg up to domain_size_km ~1500"
CONVERGENCE_BENCHMARK_DEG = 0.3
CONVERGENCE_BENCHMARK_KM = 1500

# -------- real-data comparison: edit these to look at a different file/time --------
# using nemo_forcing (ERA5 wind) rather than nemo_ocean current: current is noisy
# at small scales, which made two different sample points of the same field look
# deceptively different for reasons unrelated to alpha_deg
DATASET = "nemo_forcing"
DOMAIN = "nordic_seas"
FILE_INDEX = 0          # which sorted source file (per file_match token) to read
TIME_INDEX = 0           # which timestep inside that file to plot
PAIR_VARS = ("u10", "v10")
N_ARROWS = 9              # arrows per axis, kept sparse -- see plot_real_comparison
# arrow colour is deliberately different from contour colour -- coloured arrows get
# lost once contour lines are dense
CONTOUR_COLOR_BASELINE = "tab:blue"
CONTOUR_COLOR_CONFIGURED = "tab:orange"
CONTOUR_STYLE_BASELINE = "-"
CONTOUR_STYLE_CONFIGURED = "--"  # dashed, so an exact overlap with the solid line still shows
ARROW_COLOR_BASELINE = "black"
ARROW_COLOR_CONFIGURED = "gray"
STREAM_GRID_N = 120  # resolution of the shared mesh streamlines get interpolated onto
# -------------------------------------------------------------------------------------


def _proj_factors(lat_0, lon_0, lon_grid, lat_grid):
    ''' Same PROJ factors create_local_metric_grid computes internally, plus
    angular_distortion: how far this point is from conformal, i.e. how much
    error the rotation-only approximation leaves behind. A large
    meridian_convergence isn't itself a problem -- cos_g/sin_g already
    correct for it exactly; it's just naturally large near the pole. '''
    p = Proj(f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m")
    return p.get_factors(lon_grid, lat_grid)


def _unrotate(u_rot, v_rot, cos_g, sin_g):
    ''' undo _rotate_vectors: grid-local components back to true east/north '''
    u = u_rot * cos_g + v_rot * sin_g
    v = -u_rot * sin_g + v_rot * cos_g
    return u, v


def check_invertibility(grid, alpha_deg):
    ''' Check 1: rotate a synthetic due-east wind into the grid's local basis
    and back. Should recover the original vector almost exactly, whatever
    alpha_deg is. '''
    shape = grid["lat"].shape
    u_true = np.full(shape, 10.0)  # 10 m/s due east
    v_true = np.zeros(shape)

    ds = xr.Dataset({
        "u": (("y", "x"), u_true.copy()),
        "v": (("y", "x"), v_true.copy()),
    })
    rotated = _rotate_vectors(ds, ("u", "v"), grid)
    u_rot, v_rot = rotated["u"].values, rotated["v"].values

    u_back, v_back = _unrotate(u_rot, v_rot, grid["cos_g"].values, grid["sin_g"].values)

    max_component_error = max(np.abs(u_back - u_true).max(), np.abs(v_back - v_true).max())
    mag_true = np.hypot(u_true, v_true)
    mag_rot = np.hypot(u_rot, v_rot)
    max_magnitude_error = np.abs(mag_rot - mag_true).max()

    print(f"alpha_deg={alpha_deg:>6.2f}: "
          f"max component round-trip error = {max_component_error:.2e} m/s, "
          f"max magnitude change from rotation = {max_magnitude_error:.2e} m/s")

    return u_rot, v_rot, u_back, v_back


def check_convergence_spread(grid, alpha_deg, lat_0, lon_0):
    ''' Check 2: how much error does the rotation-only approximation actually
    carry at this domain's size? Uses angular_distortion, not raw meridian
    convergence -- convergence is just the (correctly applied) rotation
    angle, and is naturally large near the pole regardless of any error. '''
    factors = _proj_factors(lat_0, lon_0, grid["lon"], grid["lat"])
    max_convergence = np.abs(factors.meridian_convergence).max()
    max_distortion = np.abs(factors.angular_distortion).max()
    within_benchmark = max_distortion <= CONVERGENCE_BENCHMARK_DEG
    print(
        f"alpha_deg={alpha_deg:>6.2f}: max |meridian convergence| = "
        f"{max_convergence:.2f} deg (expected -- large near the pole, and fully "
        f"corrected by cos_g/sin_g); max |angular distortion| (rotation-only "
        f"approximation error) = {max_distortion:.3f} deg "
        f"({'within' if within_benchmark else 'somewhat above'} the "
        f"{CONVERGENCE_BENCHMARK_DEG} deg / {CONVERGENCE_BENCHMARK_KM}km benchmark)"
    )
    return factors


def _to_plain(value):
    ''' convert an OmegaConf node to a plain python object; pass through anything else '''
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _load_cfg():
    # config_path is relative to this file (tests/) -- configs/ is one level up
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=[f"dataset={DATASET}", f"domain={DOMAIN}"])


def _select_files(cfg):
    ''' one file per file_match token, picking FILE_INDEX from each token's
    sorted matches (same convention as regridder.py/domain_vis.py) '''
    data_path = Path(cfg.dataset.folder)
    file_ext = cfg.dataset.get("file_ext", ".nc")
    tokens = _to_plain(cfg.domain.file_match.get(cfg.dataset.name)) or [""]
    files = []
    for tok in tokens:
        matches = sorted(data_path.glob(("*" + tok + "*" if tok else "*") + file_ext))
        files.append(matches[FILE_INDEX])
    return files


def regrid_true_vectors(cfg, ds_list, time_mask, alpha_deg):
    ''' regrid onto an alpha_deg target grid, same as the real pipeline
    (regridder.py::PreProcessing), then un-rotate PAIR_VARS back to true
    east/north so runs at different alpha_deg can be compared directly '''
    target_grid = create_local_metric_grid(
        domain_size_km=cfg.domain.domain_size, grid_size=cfg.domain.grid_size,
        lat_0=cfg.domain.lat_0, lon_0=cfg.domain.lon_0, alpha_deg=alpha_deg,
    )
    pipeline = RegridPipeline(
        target_grid=target_grid,
        variable_names=_to_plain(cfg.dataset.variable_names),
        interp_method=_to_plain(cfg.dataset.interp_method),
        extrap_method=_to_plain(cfg.dataset.extrap_method),
        pair_vars_list=_to_plain(cfg.dataset.get("pair_vars_list", [])),
        use_mask=bool(cfg.dataset.get("use_mask", True)),
    )
    ds = pipeline(ds_list, time_mask)
    u_var, v_var = PAIR_VARS
    # drop the size-1 time dim (time_mask always selects exactly one step)
    u_rot, v_rot = ds[u_var].values[0], ds[v_var].values[0]
    u_true, v_true = _unrotate(u_rot, v_rot, target_grid["cos_g"].values, target_grid["sin_g"].values)
    return target_grid, u_true, v_true


def _regular_display_mesh(lon_bounds, lat_bounds, n):
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds
    return np.meshgrid(np.linspace(lon_min, lon_max, n), np.linspace(lat_min, lat_max, n))


def _interp_to_mesh(lon, lat, values, mesh_lon, mesh_lat):
    ''' interpolate onto a shared regular lon/lat mesh -- streamplot needs even
    1-D axes, unlike quiver/contour, which accept the grid's own curvilinear
    lon/lat directly '''
    points = np.column_stack([lon.ravel(), lat.ravel()])
    good = np.isfinite(values.ravel())
    out = griddata(points[good], values.ravel()[good], (mesh_lon, mesh_lat), method="linear")
    nearest = griddata(points[good], values.ravel()[good], (mesh_lon, mesh_lat), method="nearest")
    return np.where(np.isnan(out), nearest, out)


def plot_real_comparison(cfg, out_path):
    ''' Regrid one real timestep twice (alpha_deg=0 and the configured value),
    un-rotate both back to true east/north, and plot them together: matching
    -colour speed contours plus arrows on top, streamlines below for
    comparison. If alpha_deg preserves the field, both colours should trace
    the same pattern.

    Arrows are sparse on purpose: packing many arrows along a curved grid
    makes identical-angle arrows look like they're bending to follow the
    curve. Contours don't have that problem, so they use the full grid. '''
    files = _select_files(cfg)
    print("files:", [str(f) for f in files])

    reader_fn = instantiate(cfg.dataset.reader_fn)
    ds_list = reader_fn(files)

    n_time = ds_list[0].sizes["time"]
    time_mask = np.zeros(n_time, dtype=bool)
    time_mask[TIME_INDEX] = True
    print("time:", np.asarray(ds_list[0]["time"].values[TIME_INDEX]))

    alpha_configured = float(cfg.domain.get("alpha_deg", 0.0))
    runs = [
        {
            "alpha_deg": 0.0, "label": "alpha_deg=0.0",
            "contour_color": CONTOUR_COLOR_BASELINE, "contour_style": CONTOUR_STYLE_BASELINE,
            "arrow_color": ARROW_COLOR_BASELINE,
        },
        {
            "alpha_deg": alpha_configured, "label": f"alpha_deg={alpha_configured}",
            "contour_color": CONTOUR_COLOR_CONFIGURED, "contour_style": CONTOUR_STYLE_CONFIGURED,
            "arrow_color": ARROW_COLOR_CONFIGURED,
        },
    ]

    fields = []
    for run in runs:
        grid, u_true, v_true = regrid_true_vectors(cfg, ds_list, time_mask, run["alpha_deg"])
        fields.append({**run, "grid": grid, "u": u_true, "v": v_true})

    all_speeds = [np.hypot(f["u"], f["v"]) for f in fields]
    for f, speed in zip(fields, all_speeds):
        p50, p95, p99 = np.nanpercentile(speed, [50, 95, 99])
        print(f"{f['label']}: speed percentiles p50={p50:.4f} p95={p95:.4f} p99={p99:.4f} "
              f"max={np.nanmax(speed):.4f} m/s")
    # use the 95th percentile for contour levels, not the max -- one sharp
    # coastal pixel would otherwise blow out the whole scale
    v95 = float(np.nanmax([np.nanpercentile(s, 95) for s in all_speeds]))
    levels = np.linspace(0, v95, 8)[1:]  # drop 0, it's just the land-mask edge

    # maps are wide/short (cartopy keeps their true aspect ratio) -- a taller
    # figure just adds blank space between panels instead of closing it
    fig = plt.figure(constrained_layout=True, figsize=(15, 11))
    ax_top = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax_bottom = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax_top.coastlines(resolution="50m")
    ax_bottom.coastlines(resolution="50m")

    # combine both runs into one quiver() call -- separate calls autoscale
    # independently, which would make arrow lengths incomparable between colours
    all_lon, all_lat, all_u, all_v, all_colors = [], [], [], [], []
    for f in fields:
        grid, u_true, v_true = f["grid"], f["u"], f["v"]
        speed = np.hypot(u_true, v_true)
        ax_top.contour(
            grid["lon"], grid["lat"], speed, levels=levels, colors=f["contour_color"],
            linestyles=f["contour_style"], linewidths=1.2, transform=ccrs.PlateCarree(),
        )

        step = max(grid["lat"].shape[0] // N_ARROWS, 1)
        lon_s = grid["lon"][::step, ::step]
        lat_s = grid["lat"][::step, ::step]
        u_s, v_s = u_true[::step, ::step], v_true[::step, ::step]
        finite = np.isfinite(u_s) & np.isfinite(v_s)
        all_lon.append(lon_s[finite])
        all_lat.append(lat_s[finite])
        all_u.append(u_s[finite])
        all_v.append(v_s[finite])
        all_colors.append(np.full(finite.sum(), f["arrow_color"], dtype=object))

    quiver_artist = ax_top.quiver(
        np.concatenate(all_lon), np.concatenate(all_lat),
        np.concatenate(all_u), np.concatenate(all_v),
        color=np.concatenate(all_colors), transform=ccrs.PlateCarree(),
        width=0.0022,  # thinner shafts -- default (0.005) is too heavy over dense contours
    )
    ax_top.quiverkey(
        quiver_artist, X=0.08, Y=-0.06, U=v95, label=f"{v95:.1f} m/s (95th percentile speed)",
        labelpos="E", coordinates="axes",
    )
    ax_top.legend(
        handles=[
            Line2D(
                [], [], color=f["arrow_color"], marker=">", linestyle=f["contour_style"],
                markersize=8, label=f"{f['label']}  (arrow={f['arrow_color']}, "
                                     f"contour {f['contour_style']!r} {f['contour_color']})",
            )
            for f in fields
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
    )
    ax_top.set_title(
        f"{cfg.dataset.name} {PAIR_VARS} at t={ds_list[0]['time'].values[TIME_INDEX]}\n"
        f"true (un-rotated) vectors + speed contours -- "
        f"alpha_deg=0.0 vs alpha_deg={alpha_configured}"
    )

    # bottom panel: same comparison as streamlines. streamplot needs a regular
    # 1-D grid, so interpolate each run onto a shared mesh first
    lon_min = min(np.nanmin(f["grid"]["lon"]) for f in fields)
    lon_max = max(np.nanmax(f["grid"]["lon"]) for f in fields)
    lat_min = min(np.nanmin(f["grid"]["lat"]) for f in fields)
    lat_max = max(np.nanmax(f["grid"]["lat"]) for f in fields)
    mesh_lon, mesh_lat = _regular_display_mesh(
        (lon_min, lon_max), (lat_min, lat_max), STREAM_GRID_N
    )
    mesh_lon_1d, mesh_lat_1d = mesh_lon[0, :], mesh_lat[:, 0]

    # matplotlib's automatic seeding starts a different number of lines per call
    # depending on how fast each field's trajectories happen to terminate --
    # that's noise, not a real alpha_deg difference. Using the same start_points
    # for both runs makes it a fair, same-starting-point comparison.
    # (inset a bit from the mesh edge -- a seed exactly on the boundary can fail
    # matplotlib's check due to floating-point rounding)
    lon_pad, lat_pad = 0.1 * (lon_max - lon_min), 0.1 * (lat_max - lat_min)
    seed_lon, seed_lat = _regular_display_mesh(
        (lon_min + lon_pad, lon_max - lon_pad), (lat_min + lat_pad, lat_max - lat_pad), 12
    )
    start_points = np.column_stack([seed_lon.ravel(), seed_lat.ravel()])

    # baseline wider/behind, configured thinner/on top -- where they overlap,
    # the wider line still peeks out instead of being fully hidden. Both are
    # semi-transparent for the same reason.
    stream_linewidths = {fields[0]["contour_color"]: 2.4, fields[1]["contour_color"]: 1.1}
    for f in fields:
        grid, u_true, v_true = f["grid"], f["u"], f["v"]
        u_grid = _interp_to_mesh(grid["lon"], grid["lat"], u_true, mesh_lon, mesh_lat)
        v_grid = _interp_to_mesh(grid["lon"], grid["lat"], v_true, mesh_lon, mesh_lat)
        # same colours as the top panel's contours, so colour still means
        # "which run" here too.
        # Calling matplotlib's streamplot directly instead of cartopy's wrapper:
        # cartopy's version rejected valid start_points (an internal regridding
        # step of its own). Our axes is already plain PlateCarree, so going
        # straight to matplotlib gives the same result without that bug.
        maxes.Axes.streamplot(
            ax_bottom, mesh_lon_1d, mesh_lat_1d, u_grid, v_grid,
            color=to_rgba(f["contour_color"], 0.8),
            linewidth=stream_linewidths[f["contour_color"]], arrowsize=1.2,
            start_points=start_points,
        )
    ax_bottom.legend(
        handles=[
            Line2D([], [], color=f["contour_color"], label=f["label"]) for f in fields
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
    )
    ax_bottom.set_title("same comparison as streamlines (interpolated onto a shared regular mesh)")

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")


def main():
    # load once and reuse -- these used to be hardcoded separately for Check 1/2
    # and had drifted out of sync with the real config
    cfg = _load_cfg()
    lat_0, lon_0 = cfg.domain.lat_0, cfg.domain.lon_0
    domain_size_km, grid_size = cfg.domain.domain_size, cfg.domain.grid_size
    alphas = (0.0, float(cfg.domain.get("alpha_deg", 0.0)))

    grids = {}
    print("=== Check 1: is the rotation exact/invertible? ===")
    for alpha_deg in alphas:
        grid = create_local_metric_grid(
            domain_size_km=domain_size_km, grid_size=grid_size,
            lat_0=lat_0, lon_0=lon_0, alpha_deg=alpha_deg,
        )
        grids[alpha_deg] = grid
        check_invertibility(grid, alpha_deg)

    print("\n=== Check 2: is the off-center rotation-only approximation still "
          "valid at this domain size? ===")
    for alpha_deg, grid in grids.items():
        check_convergence_spread(grid, alpha_deg, lat_0, lon_0)

    print("\n=== Visual check: real data, alpha_deg=0 vs configured, overlaid ===")
    plot_real_comparison(cfg, Path("figures") / "alpha_vector_check.png")


if __name__ == "__main__":
    main()
