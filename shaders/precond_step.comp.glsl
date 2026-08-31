/* vkflow — preconditioned (diagonal Gauss-Newton) optimizer step.
 *
 * "Adam done right": replace Adam's empirical v (an EMA of g²) with the EXACT
 * diagonal Gauss-Newton curvature D = diag(H_GN), computed by the data/wall/PDE
 * diag kernels. The step is a damped Jacobi-Newton update with momentum:
 *
 *   m   = β1·m + (1-β1)·g ;   m̂ = m/(1-β1^t)
 *   coef -= lr · m̂ / (D + μ)
 *
 * Because D carries the inv_width² PDE stiffness, the per-parameter curvature is
 * divided out → the stable lr is resolution-INVARIANT (≈1). μ is the LM damping
 * that regularizes near-null (gauge) directions. Grid-strided + zero-fold.
 *
 * Bindings (std430): 0 grad(in→0) · 1 m · 2 coef · 3 diag(ro) · 4 meta.
 */
#version 460

layout(local_size_x = 256) in;

layout(set = 0, binding = 4, std430) readonly buffer Meta {
    int   n_params;
    float beta1;
    float inv_bc1;     /* 1/(1 - β1^t) */
    float step_size;   /* lr */
    float mu;          /* LM damping */
} meta;

layout(set = 0, binding = 0, std430) buffer GradBuffer { float grad[]; };
layout(set = 0, binding = 1, std430) buffer MBuffer    { float m[];    };
layout(set = 0, binding = 2, std430) buffer CoefBuffer { float coef[]; };
layout(set = 0, binding = 3, std430) readonly buffer DiagBuffer { float diag[]; };

void main() {
    uint n      = uint(meta.n_params);
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride) {
        float g  = grad[i];
        float mi = meta.beta1 * m[i] + (1.0 - meta.beta1) * g;
        m[i] = mi;
        float mhat = mi * meta.inv_bc1;
        coef[i] -= meta.step_size * mhat / (diag[i] + meta.mu);
        grad[i]  = 0.0;
    }
}
