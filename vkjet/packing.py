"""Basis -> GPU table packing, shared by every kernel that gathers a jet.

The kernels take the per-dimension basis tables as four flat concatenated
buffers plus per-dimension offsets into them:

    coefs   = value polynomials          (_value_coef_concat)
    dcoefs  = first-derivative polys     (_deriv_coef_concat,  x inv_width)
    ddcoefs = second-derivative polys    (_deriv2_coef_concat, x inv_width^2)
    table   = LOOKBACK offset tables     (_table_concat)

The chain rule lives ENTIRELY here: inv_widths = n_intervals / world_extent, so
the jet slots a kernel gathers are true WORLD derivatives and one operator table
(in world units) serves every grid of a dyadic ladder.
"""
from __future__ import annotations

import os

import numpy as np

SHADER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders", "spv")
MAX_NDIM = 16


def _coef_offsets(bases):
    offs, c = [], 0
    for b in bases:
        offs.append(c); c += b.dense.size
    return offs


def _table_offsets(bases):
    offs, t = [], 0
    for b in bases:
        offs.append(t); t += b.table.size
    return offs


def _value_coef_concat(bases):
    blk = [b.dense.reshape(-1) for b in bases]
    return (np.concatenate(blk) if blk else np.zeros(0, np.float32)).astype(np.float32)


def _table_concat(bases):
    blk = [b.table.reshape(-1).astype(np.int32) for b in bases]
    return (np.concatenate(blk) if blk else np.zeros(0, np.int32)).astype(np.int32)


def _deriv_coef_concat(bases, inv_widths):
    blocks = [b.derivative(iw).dense.reshape(-1)
              for b, iw in zip(bases, inv_widths)]
    return (np.concatenate(blocks) if blocks else np.zeros(0, np.float32)).astype(np.float32)


def _deriv2_coef_concat(bases, inv_widths):
    """Second-derivative coef table (d2/dxi2 * inv_width^2)."""
    blocks = [b.derivative(iw).derivative(iw).dense.reshape(-1)
              for b, iw in zip(bases, inv_widths)]
    return (np.concatenate(blocks) if blocks else np.zeros(0, np.float32)).astype(np.float32)
