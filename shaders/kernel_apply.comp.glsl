/* gs_kernel_vulkan_naive — apply compute shader.
 *
 * Forward (adjoint == 0) and adjoint (adjoint == 1) modes share one
 * shader; the "adjoint" field in the meta buffer selects.
 *
 * Mirrors the math in gs/src/internal/kernel_math.h. Per the
 * cross-language sharing policy, this is a hand-translation, not a
 * shared header — GLSL has no pointers and the buffer-binding model
 * is structurally different from C. The cpu_naive kernel is the
 * behavioral oracle; the test suite cross-validates outputs.
 *
 * v9 Phase 3:
 *   - kv binding + decode branch dropped (apply consumes pre-encoded
 *     x only; encode lives on a separate vtable slot that takes
 *     gs_axes, not gs_tpb).
 *   - bisect / locate / encode shader functions dropped.
 *   - the integer bounds check on ix_int is dropped (NaN is the
 *     supported skip sentinel; v5 addressing wraps via positive_modulo).
 *   - new binding 5 = TableBuffer holding the concat'd int32 per-dim
 *     table data.
 *   - new per-dim Meta fields: stride, coef_period, table_period,
 *     primal_extent, table_offset (concat offset into TableBuffer).
 *
 * Bindings:
 *   set = 0:
 *     0  Meta            (readonly storage; std430 layout matches host C struct)
 *     1  XBuffer         (readonly; (n_samples, ndim) flat float)
 *     2  PrimalBuffer    (read+write; declared as uint[] so adjoint mode
 *                        can use atomicCompSwap for the float scatter)
 *     3  DualBuffer      (read+write float; forward mode writes per-thread
 *                        rows non-overlapping, no atomics needed)
 *     4  CoefsBuffer     (readonly; per-dim coefs concatenated, see
 *                        meta.coef_offset[d])
 *     5  TableBuffer     (readonly int[]; per-dim v5 tables concatenated,
 *                        see meta.table_offset[d])
 */
#version 460

#define MAX_NDIM 16
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Meta {
    int ndim;
    int n_samples;
    int n_channels;
    int num_combos;
    int adjoint;
    int _pad0;
    int _pad1;
    int _pad2;
    int coef_period  [MAX_NDIM];
    int order        [MAX_NDIM];
    int degp1        [MAX_NDIM];
    int stride       [MAX_NDIM];
    int table_period [MAX_NDIM];
    int primal_extent[MAX_NDIM];
    int pstride      [MAX_NDIM];
    int coef_offset  [MAX_NDIM];
    int table_offset [MAX_NDIM];
} meta;

layout(set = 0, binding = 1, std430) readonly buffer XBuffer      { float x[];          };
layout(set = 0, binding = 2, std430)          buffer PrimalBuffer { uint  primalU[];    };
layout(set = 0, binding = 3, std430)          buffer DualBuffer   { float dual[];       };
layout(set = 0, binding = 4, std430) readonly buffer CoefsBuffer  { float coefs[];      };
layout(set = 0, binding = 5, std430) readonly buffer TableBuffer  { int   table_data[]; };

float horner(int p_off, int degree, float xv) {
    float r = coefs[p_off + degree];
    for (int k = degree - 1; k >= 0; k--) {
        r = r * xv + coefs[p_off + k];
    }
    return r;
}

/* Non-negative remainder for signed `a`, positive `b`. Matches
 * gs_positive_modulo in kernel_math.h. GLSL's `%` is defined to follow
 * the sign of the dividend, so the `r < 0 ? r + b : r` correction is
 * needed for negative `a` (e.g. textbook N_c table = [-(order-1)..0]). */
int positive_modulo(int a, int b) {
    int r = a % b;
    return (r < 0) ? (r + b) : r;
}

/* atomicCompSwap loop for float-on-uint storage. Equivalent to
 * atomicAdd(float, float) but works without VK_EXT_shader_atomic_float. */
void atomicAddFloat(uint idx, float val) {
    uint expected = primalU[idx];
    while (true) {
        float current = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(current + val);
        uint  actual  = atomicCompSwap(primalU[idx], expected, desired);
        if (actual == expected) break;
        expected = actual;
    }
}

void main() {
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;

    int   ix_int [MAX_NDIM];
    float ix_frac[MAX_NDIM];
    bool  skip = false;

    for (int d = 0; d < meta.ndim; d++) {
        float xv = x[sn * uint(meta.ndim) + uint(d)];
        if (isnan(xv)) { skip = true; break; }
        float ipart = floor(xv);
        ix_int[d]  = int(ipart);
        ix_frac[d] = xv - ipart;
    }
    if (skip) return;

    for (int i = 0; i < meta.num_combos; i++) {
        int   reduce = i;
        int   prim   = 0;
        float weight = 1.0;
        for (int d = 0; d < meta.ndim; d++) {
            int order_d  = meta.order[d];
            int degp1_d  = meta.degp1[d];
            int idx      = reduce % order_d;
            reduce      /= order_d;

            int coef_row  = positive_modulo(ix_int[d], meta.coef_period [d]);
            int table_row = positive_modulo(ix_int[d], meta.table_period[d]);
            int t_off     = meta.table_offset[d] + table_row * order_d + idx;
            int sum_d     = ix_int[d] * meta.stride[d] + table_data[t_off];
            int wrap_di   = positive_modulo(sum_d, meta.primal_extent[d]);
            prim         += wrap_di * meta.pstride[d];

            int p_off     = meta.coef_offset[d] + (coef_row * order_d + idx) * degp1_d;
            weight       *= horner(p_off, degp1_d - 1, ix_frac[d]);
        }

        uint dual_base   = sn * uint(meta.n_channels);
        uint primal_base = uint(prim);

        if (meta.adjoint == 0) {
            /* dual += primal * weight; per-thread row, no atomics needed. */
            for (int ch = 0; ch < meta.n_channels; ch++) {
                float pv = uintBitsToFloat(primalU[primal_base + uint(ch)]);
                dual[dual_base + uint(ch)] += pv * weight;
            }
        } else {
            /* primal += dual * weight; cross-thread scatter — atomic. */
            for (int ch = 0; ch < meta.n_channels; ch++) {
                float dv = dual[dual_base + uint(ch)];
                atomicAddFloat(primal_base + uint(ch), dv * weight);
            }
        }
    }
}
