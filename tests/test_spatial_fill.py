'''Tests for spatial_fill.GapFiller: nearest/laplace gap filling on the target grid'''

import numpy as np
import pytest

import spatial_fill
from spatial_fill import GapFiller


def _ramp(ny=6, nx=9):
    ''' field linear in x: 0, 1, 2, ... per column '''
    return np.tile(np.arange(nx, dtype=np.float64), (ny, 1))


@pytest.mark.parametrize("method", ["nearest", "laplace"])
def test_valid_cells_never_change_and_land_stays_nan(method):
    values = _ramp()
    values[2:4, 3:6] = np.nan  # gap
    land = np.zeros(values.shape, dtype=bool)
    land[0, :] = True
    values[land] = np.nan

    filled = GapFiller(land, method)(values)

    valid = np.isfinite(values)
    np.testing.assert_array_equal(filled[valid], values[valid])
    assert np.isnan(filled[land]).all()
    assert np.isfinite(filled[~land]).all()


def test_nearest_copies_nearest_valid_cell():
    values = _ramp()
    values[:, 6:] = np.nan  # gap on the right edge
    filled = GapFiller(None, "nearest")(values)
    np.testing.assert_array_equal(filled[:, 6:], 5.0)  # last valid column


def test_laplace_is_linear_between_two_fixed_sides():
    ''' a gap band spanning full rows, between two valid columns, with no-flux top and
    bottom edges, has the exact linear solution -- so a ramp is recovered exactly '''
    truth = _ramp()
    values = truth.copy()
    values[:, 2:7] = np.nan
    filled = GapFiller(None, "laplace")(values)
    np.testing.assert_allclose(filled, truth, atol=1e-10)


def test_laplace_falls_back_to_nearest_for_cut_off_basin():
    ''' an ocean pocket enclosed by land, with no valid cell, can't be solved '''
    values = _ramp()
    land = np.zeros(values.shape, dtype=bool)
    land[1:5, 1] = land[1:5, 5] = land[1, 1:6] = land[4, 1:6] = True
    values[land] = np.nan
    values[2:4, 2:5] = np.nan  # the enclosed pocket

    filled = GapFiller(land, "laplace")(values)

    assert np.isfinite(filled[2:4, 2:5]).all()


@pytest.fixture
def plan_builds(monkeypatch):
    ''' counts _nearest_plan calls '''
    builds = []
    original = spatial_fill._nearest_plan

    def counting(valid, gap):
        builds.append(gap.sum())
        return original(valid, gap)
    monkeypatch.setattr(spatial_fill, "_nearest_plan", counting)
    return builds


def test_all_nan_step_is_left_missing_and_one_plan_per_pattern(plan_builds):
    filler = GapFiller(None, "nearest")
    a = _ramp()
    a[:, :2] = np.nan
    b = _ramp()
    b[:, -2:] = np.nan
    steps = np.stack([a, np.full_like(a, np.nan), b, a, _ramp()]).astype(np.float32)

    filled = filler(steps)

    assert filled.dtype == np.float32
    assert np.isnan(filled[1]).all()  # nothing to fill from
    assert np.isfinite(filled[[0, 2, 3, 4]]).all()
    np.testing.assert_array_equal(filled[0], filled[3])
    # only the two partial gaps need a plan, and steps 0 and 3 share one
    assert len(plan_builds) == 2


def test_last_plan_is_reused_by_the_next_call(plan_builds):
    filler = GapFiller(None, "nearest")
    a = _ramp()
    a[:, :2] = np.nan
    b = _ramp()
    b[:, -2:] = np.nan

    filler(a[None])
    filler(np.stack([a, a]))  # same pattern: reused
    assert len(plan_builds) == 1
    filler(b[None])
    filler(a[None])  # only the last plan is kept: rebuilt
    assert len(plan_builds) == 3


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        GapFiller(None, "nearest_s2d")
