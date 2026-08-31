"""Tensor-product spline ``apply`` (forward evaluation) on the GPU.

Ports the host-side glue of gs's ``kernel_vulkan_naive`` apply: packs the
``Meta`` std430 block, concatenates per-dim coefficient + table buffers with
offsets, and dispatches ``kernel_apply.spv`` (one thread per sample) through
the vkjet harness.

Meta layout (std430, matches kernel_apply.comp.glsl exactly):

    int ndim, n_samples, n_channels, num_combos, adjoint, _pad0, _pad1, _pad2;
    int coef_period[16], order[16], degp1[16], stride[16], table_period[16],
        primal_extent[16], pstride[16], coef_offset[16], table_offset[16];

i.e. 8 scalar ints + 9 int[16] arrays = 152 int32 (std430 scalar arrays are
tightly packed at 4-byte stride). All six bindings are storage buffers.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from .context import Context, STORAGE
from .basis import Basis1D

MAX_NDIM = 16
_HEADER_INTS = 8
_NUM_ARRAYS = 9
_META_INTS = _HEADER_INTS + _NUM_ARRAYS * MAX_NDIM  # 152

from .packing import SHADER_DIR
APPLY_SPV = os.path.join(SHADER_DIR, "kernel_apply.spv")


def primal_strides(extents: Sequence[int], n_channels: int) -> List[int]:
    """Row-major strides of a primal grid (e0,...,e_{nd-1}, n_channels)."""
    nd = len(extents)
    pstride = [0] * nd
    acc = n_channels
    for d in range(nd - 1, -1, -1):
        pstride[d] = acc
        acc *= extents[d]
    return pstride


def pack_apply_meta(bases: Sequence[Basis1D], n_channels: int, n_samples: int,
                    adjoint: int = 0):
    """Return (meta_bytes, coefs_f32, table_i32, pstride) for the apply kernel."""
    nd = len(bases)
    assert nd <= MAX_NDIM
    order = [b.order for b in bases]
    num_combos = int(np.prod(order))
    extents = [b.primal_extent for b in bases]
    pstride = primal_strides(extents, n_channels)

    # concat per-dim coefficient (float) and table (int) blocks; record offsets
    coef_blocks, table_blocks = [], []
    coef_offset, table_offset = [], []
    c_off = t_off = 0
    for b in bases:
        coef_offset.append(c_off)
        table_offset.append(t_off)
        cb = b.dense.reshape(-1)              # (coef_period*order*degp1,)
        tb = b.table.reshape(-1).astype(np.int32)  # (table_period*order,)
        coef_blocks.append(cb)
        table_blocks.append(tb)
        c_off += cb.size
        t_off += tb.size
    coefs_f32 = (np.concatenate(coef_blocks) if coef_blocks
                 else np.zeros(0, np.float32)).astype(np.float32)
    table_i32 = (np.concatenate(table_blocks) if table_blocks
                 else np.zeros(0, np.int32)).astype(np.int32)

    m = np.zeros(_META_INTS, dtype=np.int32)
    m[0] = nd; m[1] = n_samples; m[2] = n_channels; m[3] = num_combos; m[4] = adjoint

    def put(slot_index, values):
        base = _HEADER_INTS + slot_index * MAX_NDIM
        m[base:base + len(values)] = values

    put(0, [b.coef_period for b in bases])
    put(1, order)
    put(2, [b.degp1 for b in bases])
    put(3, [b.stride for b in bases])
    put(4, [b.table_period for b in bases])
    put(5, extents)
    put(6, pstride)
    put(7, coef_offset)
    put(8, table_offset)
    return m.tobytes(), coefs_f32, table_i32, pstride


class ApplyForward:
    """Forward tensor-product apply: dual[s,ch] = Σ_combos w · primal[prim+ch]."""

    def __init__(self, ctx: Context, bases: Sequence[Basis1D], n_channels: int):
        self.ctx = ctx
        self.bases = list(bases)
        self.nd = len(bases)
        self.n_channels = n_channels
        self.extents = [b.primal_extent for b in bases]
        self.primal_count = int(np.prod(self.extents)) * n_channels
        self.program = ctx.program(APPLY_SPV, bindings=[STORAGE] * 6)

    def forward(self, x_enc: np.ndarray, primal: np.ndarray) -> np.ndarray:
        x_enc = np.ascontiguousarray(x_enc, dtype=np.float32)
        n_samples = x_enc.shape[0]
        assert x_enc.shape[1] == self.nd
        primal = np.ascontiguousarray(primal, dtype=np.float32).reshape(-1)
        assert primal.size == self.primal_count, \
            f"primal size {primal.size} != {self.primal_count}"

        meta_b, coefs, table, _ = pack_apply_meta(
            self.bases, self.n_channels, n_samples, adjoint=0)

        ctx = self.ctx
        meta_buf = ctx.buffer(len(meta_b), device_local=False); meta_buf.upload(meta_b)
        x_buf = ctx.buffer(x_enc.nbytes); x_buf.upload(x_enc.reshape(-1))
        primal_buf = ctx.buffer(primal.nbytes); primal_buf.upload(primal)
        dual_buf = ctx.buffer(n_samples * self.n_channels * 4); dual_buf.zero()
        coefs_buf = ctx.buffer(max(coefs.nbytes, 4)); coefs_buf.upload(coefs)
        table_buf = ctx.buffer(max(table.nbytes, 4)); table_buf.upload(table)

        groups = (n_samples + 255) // 256   # local_size_x = 256
        ctx.run(self.program,
                [meta_buf, x_buf, primal_buf, dual_buf, coefs_buf, table_buf],
                groups=groups)
        return dual_buf.download(np.float32, n_samples * self.n_channels) \
                       .reshape(n_samples, self.n_channels)


# --------------------------------------------------------------------------- #
# Pure-numpy oracle (independent ground truth) — the v5 combo math.
# --------------------------------------------------------------------------- #
def apply_oracle(bases: Sequence[Basis1D], x_enc: np.ndarray,
                 primal: np.ndarray, n_channels: int) -> np.ndarray:
    import itertools
    nd = len(bases)
    n_samples = x_enc.shape[0]
    extents = [b.primal_extent for b in bases]
    order = [b.order for b in bases]
    primal = primal.reshape(tuple(extents) + (n_channels,))
    out = np.zeros((n_samples, n_channels), dtype=np.float64)
    for s in range(n_samples):
        ix_int = [int(np.floor(x_enc[s, d])) for d in range(nd)]
        ix_frac = [float(x_enc[s, d] - np.floor(x_enc[s, d])) for d in range(nd)]
        for offsets in itertools.product(*[range(o) for o in order]):
            w = 1.0
            idxs = []
            for d in range(nd):
                b = bases[d]
                coef_row = ix_int[d] % b.coef_period
                table_row = ix_int[d] % b.table_period
                sum_d = ix_int[d] * b.stride + int(b.table[table_row, offsets[d]])
                wrap = sum_d % b.primal_extent
                poly = b.dense[coef_row, offsets[d], :]
                w *= float(np.polynomial.polynomial.polyval(ix_frac[d], poly))
                idxs.append(wrap)
            out[s, :] += primal[tuple(idxs)] * w
    return out
