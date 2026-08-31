/* vkflow — LM trial step: coef = backup − lr·grad/(diag + μ).
 *
 * Reads from a coef BACKUP and writes coef, leaving grad/momentum untouched so
 * the LM accept/reject loop can retry the same gradient at different μ. No
 * momentum (LM uses β1=0; the curvature provides the smoothing).
 *
 * Bindings: 0 coef(out) · 1 backup(ro) · 2 grad(ro) · 3 diag(ro) · 4 meta.
 */
#version 460
layout(local_size_x = 256) in;

layout(set = 0, binding = 4, std430) readonly buffer Meta {
    int   n_params;
    float lr;
    float mu;
} meta;
layout(set = 0, binding = 0, std430)          buffer CoefBuffer   { float coef[];   };
layout(set = 0, binding = 1, std430) readonly buffer BackupBuffer { float backup[]; };
layout(set = 0, binding = 2, std430) readonly buffer GradBuffer   { float grad[];   };
layout(set = 0, binding = 3, std430) readonly buffer DiagBuffer   { float diag[];   };

void main() {
    uint n = uint(meta.n_params);
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride) {
        coef[i] = backup[i] - meta.lr * grad[i] / (diag[i] + meta.mu);
    }
}
