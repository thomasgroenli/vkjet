"""vkjet — jet-row field fitting on Vulkan.

Solve arbitrary (<= quadratic) PDE + measurement systems posed as ROWS over the
second-order jet of a multi-channel spline field, on the GPU, with kernels
generated per operator at run time.

    from vkjet import Axes, OperatorTable, make_rows, fit_rows

    ops = OperatorTable()
    k   = ops.add_op("continuity", lin=[(SLOT_DX, 0, 1.0), (SLOT_DY, 1, 1.0),
                                        (SLOT_DZ, 2, 1.0)])
    rows = make_rows(x, np.full(len(x), k), w, s)
    res  = fit_rows(rows, ops, lo=lo, hi=hi)
    u    = res.forward(x_query)

Every row is (x, op_id, w, s, c[8]); the operator table gives

    r = <L, J(f)(x)> + J^T Q J - s ,     loss += 1/2 * scale * w * r^2

Physics are rows: collocation points are explicit rows and a PDE is a
measurement, so lambda knobs are row weights, spatially varying enforcement is
a weight column, spatially varying coefficients ride the per-row payload, and
an experiment is a file diff.

Dependencies: volkano + numpy. The B-spline basis generator is vendored
(vkjet._gs); a GLSL compiler (glslc or glslangValidator) is optional and only
enables the JIT tier — without it everything runs on the generic kernel.
"""
from .basis import Basis1D                                        # noqa: F401
from .context import Context, Buffer, ComputeProgram, STORAGE, UNIFORM  # noqa: F401,E501
from .data import (Axes, make_rows, merge_rows, merge_row_sets,   # noqa: F401
                   save_rows, load_rows, data_operator,
                   rows_from_unified, rows_hash,
                   load_unified, save_unified)
from .eqrow import (OperatorTable, EqRowTerm, NF, NCH, NCPR,      # noqa: F401
                    SLOT_VAL, SLOT_DT, SLOT_DX, SLOT_DY, SLOT_DZ,
                    SLOT_DTT, SLOT_DXX, SLOT_DYY, SLOT_DZZ,
                    SLOT_DTX, SLOT_DTY, SLOT_DTZ,
                    SLOT_DXY, SLOT_DXZ, SLOT_DYZ,
                    SLOT_NAMES, MIXED_PAIRS)
from .optim import GaussNewtonCG                                  # noqa: F401
from .rowfit import fit_rows, FitResult, multilinear_resize       # noqa: F401

__version__ = "0.1.0"
