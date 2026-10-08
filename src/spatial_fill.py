'''
Fill NaN gaps on the target grid from the valid cells around them, after regridding.
Used by `grid_interp.RegridPipeline`. Land cells are never filled.

The target grid is uniform and metric, so grid-index distance is real distance.
'''

import hashlib
import logging

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse.linalg import splu

logger = logging.getLogger(__name__)

FILL_METHODS = ("nearest", "laplace")

# 4-neighbour stencil
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _nearest_plan(valid, gap):
    ''' copy each gap cell from its nearest valid cell '''
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    gy, gx = np.nonzero(gap)
    sy, sx = iy[gy, gx], ix[gy, gx]

    def apply(block):
        block[:, gy, gx] = block[:, sy, sx]
        return block
    return apply


def _laplace_plan(valid, gap):
    '''
    Solve Laplace's equation over the gap cells: valid cells are fixed values, land and
    the domain edge are no-flux.

    a gap region with no valid cell among its 4-neighbours (only land and/or the domain
    edge around it) has no unique solution, so it falls back to nearest
    '''
    ny, nx = gap.shape
    labels, _ = label(gap)  # 4-connected gap regions

    gy, gx = np.nonzero(gap)
    n = gy.size
    uid = np.full(gap.shape, -1)
    uid[gy, gx] = np.arange(n)

    deg = np.zeros(n)
    rows_a, cols_a, rows_c, cols_c = [], [], [], []
    for dy, dx in _NEIGHBOURS:
        yy, xx = gy + dy, gx + dx
        inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        r, yy, xx = np.flatnonzero(inside), yy[inside], xx[inside]
        is_gap, is_valid = gap[yy, xx], valid[yy, xx]
        deg[r[is_gap | is_valid]] += 1
        rows_a.append(r[is_gap])
        cols_a.append(uid[yy[is_gap], xx[is_gap]])
        rows_c.append(r[is_valid])
        cols_c.append(yy[is_valid] * nx + xx[is_valid])
    rows_a, cols_a = np.concatenate(rows_a), np.concatenate(cols_a)
    rows_c, cols_c = np.concatenate(rows_c), np.concatenate(cols_c)

    # regions touching at least one valid cell are solvable
    solvable_labels = np.unique(labels[gy[rows_c], gx[rows_c]])
    solvable = np.isin(labels[gy, gx], solvable_labels)

    # renumber unknowns to the solvable cells only (regions never share an edge)
    keep = np.full(n, -1)
    keep[solvable] = np.arange(solvable.sum())
    a_ok = solvable[rows_a]
    c_ok = solvable[rows_c]
    m = int(solvable.sum())

    lu = None
    if m:
        a = sp.diags(deg[solvable]) - sp.csr_matrix(
            (np.ones(a_ok.sum()), (keep[rows_a[a_ok]], keep[cols_a[a_ok]])), shape=(m, m),
        )
        lu = splu(a.tocsc())
    c = sp.csr_matrix(
        (np.ones(c_ok.sum()), (keep[rows_c[c_ok]], cols_c[c_ok])), shape=(m, ny * nx),
    )
    sy, sx = gy[solvable], gx[solvable]

    rest = np.zeros_like(gap)
    rest[gy[~solvable], gx[~solvable]] = True
    nearest = _nearest_plan(valid, rest) if rest.any() else None

    def apply(block):
        if lu is not None:
            flat = block.reshape(block.shape[0], -1).T.astype(np.float64)
            # NOTE: c only references valid cells, so the NaNs in `flat` are never touched
            block[:, sy, sx] = lu.solve(c @ flat).T
        return nearest(block) if nearest is not None else block
    return apply


class GapFiller:
    '''
    Fill NaN cells that aren't land, one plan per NaN pattern. Coverage can change over
    time (e.g. HBM forcing), so plans are keyed by pattern, not built once.

    Parameters
    ----------
    land_mask : (y, x) bool array or None
        True = land, never filled. None = every NaN cell is a gap.
    method : str
        "nearest" or "laplace"

    Steps with no valid cell at all are left NaN, so they stay "missing".
    '''

    def __init__(self, land_mask, method):
        if method not in FILL_METHODS:
            raise ValueError(f"fill_method must be one of {FILL_METHODS}, got {method!r}")
        self.method = method
        self.ocean = None if land_mask is None else ~np.asarray(land_mask, dtype=bool)

        # (pattern hash, plan) of the last pattern; plan is None when there's nothing to fill.
        # only the last one is kept
        self._last = (None, None)
        self._logged = False

    def _build_plan(self, valid):
        gap = ~valid if self.ocean is None else self.ocean & ~valid
        if not gap.any() or not valid.any():
            return None

        # plans are rebuilt often (ocean NaN changes step to step), so only log the first one
        log = logger.debug if self._logged else logger.info
        log("gap fill: filling %d cells (%s)", gap.sum(), self.method)
        self._logged = True

        plan = _nearest_plan if self.method == "nearest" else _laplace_plan
        return plan(valid, gap)

    def __call__(self, values):
        ''' values: (..., y, x) array -> filled copy, same shape and dtype '''
        out = np.array(values, copy=True)
        flat = out.reshape(-1, *out.shape[-2:])
        valid = np.isfinite(flat)

        steps = {}
        for t, step_valid in enumerate(valid):
            key = hashlib.sha1(np.packbits(step_valid)).hexdigest()
            steps.setdefault(key, []).append(t)

        for key, idx in steps.items():
            if key != self._last[0]:
                self._last = (key, self._build_plan(valid[idx[0]]))
            plan = self._last[1]
            if plan is not None:
                flat[idx] = plan(flat[idx])
        return out
