'''Tests for nemo_reader: grid-angle computation, T-point colocation, unrotation'''

import numpy as np
import xarray as xr

from nemo_reader import (
    NemoOceanReader,
    _bearing,
    _to_t_point,
    compute_t_point_angle,
    read_nemo_bathymetry,
    unrotate_to_geographic,
)

# ---- _bearing --------------------------------------------------------------

def test_bearing_due_north():
    b = _bearing(0.0, 0.0, 0.0, 1.0)
    assert np.isclose(np.rad2deg(b), 0.0, atol=1e-6)


def test_bearing_due_east():
    b = _bearing(0.0, 0.0, 1.0, 0.0)
    assert np.isclose(np.rad2deg(b), 90.0, atol=1e-6)


# ---- compute_t_point_angle ---------------------------------------------

def test_compute_t_point_angle_recovers_known_rotation():
    '''
    Build a synthetic curvilinear patch by placing every point via an exact
    geodesic (pyproj.Geod.fwd -- bearing + distance from a common center, no
    flat-plane/small-angle approximation) so that every point's offset from the
    center has a precisely known compass bearing (grid_bearing + a known
    rotation `alpha`), then confirm compute_t_point_angle recovers `alpha` (an
    independent construction, not just a self-consistency check).

    Uses a spherical geodesic (matching `_bearing`'s own spherical formula --
    an ellipsoidal one would introduce a small systematic sphere-vs-ellipsoid
    bias unrelated to `compute_t_point_angle`'s correctness) and a small
    domain (100m spacing) so meridian convergence -- the local compass bearing
    between two nearby points drifting from the nominal bearing-from-center as
    you move away from the construction center, the same real effect
    `create_local_metric_grid`'s `cos_g`/`sin_g` correct for -- stays far below
    the test's tolerance; neither is an approximation error in
    `compute_t_point_angle` itself.
    '''
    from pyproj import Geod  # pylint: disable=import-outside-toplevel

    alpha = np.deg2rad(25.0)
    lat0, lon0 = 60.0, 10.0
    spacing_m = 100.0
    ny, nx = 5, 5
    geod = Geod(ellps="sphere")

    def place(i_off, j_off):
        ''' point at local rotated-grid offset (i_off, j_off) meters from (lat0, lon0):
        equivalent to moving distance=hypot(i_off,j_off) at bearing=alpha+atan2(i_off,j_off)
        from the center, since rotating a 2-D offset by alpha just adds alpha to its bearing '''
        distance = np.hypot(i_off, j_off)
        bearing = np.rad2deg(alpha) + np.rad2deg(np.arctan2(i_off, j_off))
        lon, lat, _ = geod.fwd(
            np.full_like(distance, lon0), np.full_like(distance, lat0), bearing, distance,
        )
        return lon, lat

    i_idx, j_idx = np.meshgrid(np.arange(nx), np.arange(ny))
    i_off = (i_idx - (nx - 1) / 2) * spacing_m
    j_off = (j_idx - (ny - 1) / 2) * spacing_m

    glamt, gphit = place(i_off, j_off)
    glamv, gphiv = place(i_off, j_off + 0.5 * spacing_m)

    domain_cfg = xr.Dataset({
        "glamt": (("y", "x"), glamt), "gphit": (("y", "x"), gphit),
        "glamv": (("y", "x"), glamv), "gphiv": (("y", "x"), gphiv),
    })

    gcost, gsint = compute_t_point_angle(domain_cfg)

    # skip j=0 (one-sided edge, less accurate)
    assert np.allclose(gcost[1:, :], np.cos(alpha), atol=1e-4)
    assert np.allclose(gsint[1:, :], np.sin(alpha), atol=1e-4)


# ---- _to_t_point -------------------------------------------------------

def test_to_t_point_nan_aware_and_one_sided_edge():
    u = xr.DataArray(np.array([[1.0, 3.0, np.nan, np.nan]]), dims=("y", "x"))
    v = xr.DataArray(np.array([[1.0], [3.0], [np.nan], [np.nan]]), dims=("y", "x"))

    u_t, v_t = _to_t_point(u, v)

    assert np.allclose(u_t.values, [[1.0, 2.0, 3.0, np.nan]], equal_nan=True)
    assert np.allclose(v_t.values, [[1.0], [2.0], [3.0], [np.nan]], equal_nan=True)


# ---- unrotate_to_geographic ---------------------------------------------

def test_unrotate_identity_at_zero_angle():
    u_e, v_n = unrotate_to_geographic(2.0, -1.0, gcost=1.0, gsint=0.0)
    assert np.isclose(u_e, 2.0)
    assert np.isclose(v_n, -1.0)


def test_unrotate_known_angle_matches_bearing_definition():
    # grid rotated so the local j-axis points at compass bearing 30deg;
    # a pure +j vector should have true (east,north) = (sin(30), cos(30))
    alpha = np.deg2rad(30.0)
    gcost, gsint = np.cos(alpha), np.sin(alpha)

    u_e, v_n = unrotate_to_geographic(0.0, 1.0, gcost, gsint)
    assert np.isclose(u_e, np.sin(alpha))
    assert np.isclose(v_n, np.cos(alpha))


def test_unrotate_is_transpose_of_rotate_vectors():
    ''' unrotate_to_geographic (local->true) should invert grid_interp._rotate_vectors
    (true->local) at the same angle -- round trip returns the original vector. '''
    from grid_interp import _rotate_vectors  # pylint: disable=import-outside-toplevel

    alpha = np.deg2rad(17.0)
    gcost, gsint = np.cos(alpha), np.sin(alpha)
    target_grid = {"cos_g": xr.DataArray(gcost), "sin_g": xr.DataArray(gsint)}

    u_true, v_true = 2.0, -3.0
    u_e, v_n = unrotate_to_geographic(u_true, v_true, gcost, gsint)

    ds = xr.Dataset({"u": xr.DataArray(u_e), "v": xr.DataArray(v_n)})
    rotated_back = _rotate_vectors(ds, ("u", "v"), target_grid)

    assert np.isclose(float(rotated_back["u"]), u_true)
    assert np.isclose(float(rotated_back["v"]), v_true)


# ---- NemoOceanReader -----------------------------------------------------

def _synthetic_files(tmp_path, ny=3, nx=3):
    ''' small synthetic ssh/ubar/vbar files + domain_cfg, no real NEMO data needed '''
    lat0, lon0 = 60.0, 10.0
    lat = lat0 + 0.1 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = lon0 + 0.1 * np.arange(nx)[None, :] * np.ones((ny, 1))
    time_counter = np.array(["2000-01-01T00:00", "2000-01-01T01:00"], dtype="datetime64[ns]")

    domain_cfg = xr.Dataset({
        "glamt": (("y", "x"), lon), "gphit": (("y", "x"), lat),
        "glamu": (("y", "x"), lon + 0.05), "gphiu": (("y", "x"), lat),
        "glamv": (("y", "x"), lon), "gphiv": (("y", "x"), lat + 0.05),
        "top_level": (("y", "x"), np.ones((ny, nx), dtype=int)),  # all-ocean by default
    })
    domain_cfg_path = tmp_path / "domain_cfg.nc"
    domain_cfg.to_netcdf(domain_cfg_path)

    rng = np.random.default_rng(0)

    def make_ds(var, extra_coords):
        data = rng.normal(size=(2, ny, nx))
        ds = xr.Dataset(
            {var: (("time_counter", "y", "x"), data)},
            coords={"time_counter": time_counter, **extra_coords},
        )
        return ds

    ssh_ds = make_ds("ssh", {"nav_lat": (("y", "x"), lat), "nav_lon": (("y", "x"), lon)})
    u_ds = make_ds("ubar", {"nav_lat": (("y", "x"), lat), "nav_lon": (("y", "x"), lon + 0.05)})
    v_ds = make_ds("vbar", {"nav_lat": (("y", "x"), lat + 0.05), "nav_lon": (("y", "x"), lon)})

    ssh_path, u_path, v_path = tmp_path / "ssh.nc", tmp_path / "ubar.nc", tmp_path / "vbar.nc"
    ssh_ds.to_netcdf(ssh_path)
    u_ds.to_netcdf(u_path)
    v_ds.to_netcdf(v_path)

    return domain_cfg_path, [ssh_path, u_path, v_path]


def test_nemo_ocean_reader_output_matches_read_nc_contract(tmp_path):
    domain_cfg_path, files = _synthetic_files(tmp_path)

    reader = NemoOceanReader(domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar")
    ds_list = reader(files)

    assert len(ds_list) == 1
    ds = ds_list[0]

    assert set(ds.data_vars) == {"ssh", "ubar", "vbar", "source_mask"}
    assert "time" in ds.coords and "time_counter" not in ds.coords
    assert ds["lat"].dims == ("y", "x")
    assert ds["lon"].dims == ("y", "x")
    assert ds["ssh"].dims == ("time", "y", "x")
    # merge succeeded despite each source file's own, differently-staggered
    # nav_lat/nav_lon -- the canonical output lat/lon comes from domain_cfg's
    # T-point coords only
    with xr.open_dataset(domain_cfg_path) as domain_cfg:
        assert np.array_equal(ds["lat"].values, domain_cfg["gphit"].values)
        assert np.array_equal(ds["lon"].values, domain_cfg["glamt"].values)


def test_nemo_ocean_reader_masks_land_and_avoids_colocation_contamination(tmp_path):
    '''
    top_level is the authoritative land mask: land T-points must end up NaN even
    when the raw source files leave garbage (non-NaN) values there, as the real
    files do (checked directly against the real dataset: ssh/ubar/vbar each leave
    ~60% of domain_cfg's land cells non-NaN). And that garbage must not
    contaminate the colocated value at an adjacent ocean T-point.

    Grid: 2x3 (y, x); column 0 = land, columns 1-2 = ocean. glamv/gphiv mirror
    _synthetic_files' pattern (longitude constant along y, latitude increasing)
    so the grid angle is exactly unrotated (gcost=1, gsint=0) -- keeps the
    expected post-rotation values exact instead of needing to hand-compute a
    rotation.
    '''
    ny, nx = 2, 3
    lat0, lon0 = 60.0, 10.0
    lat = lat0 + 0.1 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = lon0 + 0.1 * np.arange(nx)[None, :] * np.ones((ny, 1))
    top_level = np.array([[0, 1, 1], [0, 1, 1]])  # column 0 = land
    time_counter = np.array(["2000-01-01T00:00"], dtype="datetime64[ns]")

    domain_cfg = xr.Dataset({
        "glamt": (("y", "x"), lon), "gphit": (("y", "x"), lat),
        "glamu": (("y", "x"), lon + 0.05), "gphiu": (("y", "x"), lat),
        "glamv": (("y", "x"), lon), "gphiv": (("y", "x"), lat + 0.05),
        "top_level": (("y", "x"), top_level),
    })
    domain_cfg_path = tmp_path / "domain_cfg.nc"
    domain_cfg.to_netcdf(domain_cfg_path)

    garbage = 999.0  # land value the raw file leaves behind instead of NaN
    ubar_vals = np.array([[garbage, 2.0, 50.0], [garbage, 6.0, 50.0]])[None, ...]
    vbar_vals = np.zeros((1, ny, nx))
    ssh_vals = np.array([[garbage, 1.5, 2.5], [garbage, 3.5, 4.5]])[None, ...]

    def make_ds(var, values):
        return xr.Dataset(
            {var: (("time_counter", "y", "x"), values)},
            coords={
                "time_counter": time_counter,
                "nav_lat": (("y", "x"), lat), "nav_lon": (("y", "x"), lon),
            },
        )

    ssh_path, u_path, v_path = tmp_path / "ssh.nc", tmp_path / "ubar.nc", tmp_path / "vbar.nc"
    make_ds("ssh", ssh_vals).to_netcdf(ssh_path)
    make_ds("ubar", ubar_vals).to_netcdf(u_path)
    make_ds("vbar", vbar_vals).to_netcdf(v_path)

    reader = NemoOceanReader(domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar")
    ds = reader([ssh_path, u_path, v_path])[0]

    # land T-point (column 0): NaN in every variable, despite raw garbage values
    assert np.isnan(ds["ssh"].values[0, :, 0]).all()
    assert np.isnan(ds["ubar"].values[0, :, 0]).all()

    # ocean T-point at column 1 (adjacent to land column 0): west U-neighbor
    # (column 0) is masked before colocation, so this must equal the single
    # valid east neighbor (2.0 / 6.0) exactly -- not an average with `garbage`
    assert np.allclose(ds["ubar"].values[0, :, 1], [2.0, 6.0])

    # ocean T-point at column 2 (its own east U-neighbor is the masked domain
    # edge): must equal the single valid west neighbor (column 1's U-point)
    assert np.allclose(ds["ubar"].values[0, :, 2], [2.0, 6.0])

    # ssh at ocean columns passes through unaffected
    assert np.allclose(ds["ssh"].values[0, :, 1:], [[1.5, 2.5], [3.5, 4.5]])

    # embedded source_mask is domain_cfg's top_level directly, not derived
    # from any variable's own NaN pattern -- see grid_interp._create_masks
    assert np.array_equal(ds["source_mask"].values, top_level.astype(bool))


def test_nemo_ocean_reader_accepts_ssh_only(tmp_path):
    domain_cfg_path, files = _synthetic_files(tmp_path)
    ssh_path = files[0]

    reader = NemoOceanReader(domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar")
    ds = reader([ssh_path])[0]

    assert set(ds.data_vars) == {"ssh", "source_mask"}
    assert "time" in ds.coords


def test_nemo_ocean_reader_accepts_uv_only(tmp_path):
    domain_cfg_path, files = _synthetic_files(tmp_path)
    _, u_path, v_path = files

    reader = NemoOceanReader(domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar")
    ds = reader([u_path, v_path])[0]  # order shouldn't matter -- matched by name, not position

    assert set(ds.data_vars) == {"ubar", "vbar", "source_mask"}


def test_nemo_ocean_reader_rejects_unpaired_u_without_v(tmp_path):
    domain_cfg_path, files = _synthetic_files(tmp_path)
    _, u_path, _ = files

    reader = NemoOceanReader(domain_cfg_path, ssh_var="ssh", u_var="ubar", v_var="vbar")
    try:
        reader([u_path])
        assert False, "expected ValueError for ubar given without vbar"
    except ValueError:
        pass


# ---- read_nemo_bathymetry ------------------------------------------------

def test_read_nemo_bathymetry_masks_land_and_uses_t_point_coords(tmp_path):
    ''' land (top_level==0) must end up NaN even though domain_cfg leaves a literal
    0.0 there, not NaN -- same gap as ssh/ubar/vbar (see NemoOceanReader). '''
    lat0, lon0 = 60.0, 10.0
    ny, nx = 2, 3
    lat = lat0 + 0.1 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = lon0 + 0.1 * np.arange(nx)[None, :] * np.ones((ny, 1))
    top_level = np.array([[0, 1, 1], [0, 1, 1]])  # column 0 = land
    bathy = np.array([[0.0, 50.0, 100.0], [0.0, 60.0, 120.0]], dtype=np.float32)

    domain_cfg = xr.Dataset({
        "glamt": (("y", "x"), lon), "gphit": (("y", "x"), lat),
        "top_level": (("y", "x"), top_level),
        "bathy_metry": (("y", "x"), bathy),
    })
    domain_cfg_path = tmp_path / "domain_cfg.nc"
    domain_cfg.to_netcdf(domain_cfg_path)

    ds = read_nemo_bathymetry([domain_cfg_path])[0]

    assert set(ds.data_vars) == {"bathy_metry", "source_mask"}
    assert ds["bathy_metry"].dims == ("y", "x")
    assert np.array_equal(ds["lat"].values, lat)
    assert np.array_equal(ds["lon"].values, lon)

    assert np.isnan(ds["bathy_metry"].values[:, 0]).all()  # land column stays NaN
    assert np.allclose(ds["bathy_metry"].values[:, 1:], bathy[:, 1:])  # ocean unaffected

    # embedded source_mask is domain_cfg's top_level directly, same as NemoOceanReader
    assert np.array_equal(ds["source_mask"].values, top_level.astype(bool))
