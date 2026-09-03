/* vkflow — 1-D CSR factor applied along ONE axis of a flattened (outer, n_in, inner)
   array:  y[(o·n_out + j)·inner + i] = Σ_{k=ptr[j]}^{ptr[j+1]} val[k]·x[(o·n_in + col[k])·inner + i].

   The BPX transfers are a TENSOR PRODUCT of per-axis interpolation factors, so applying the
   four factors in sequence is the same operator as the assembled 4-D CSR (csr_matvec) at a
   fraction of the cost: 4 passes of <=2 (prolong) / <=3 (restrict) nonzeros instead of one
   pass of 2^4 = 16, and the index/weight tables are O(extent) instead of O(16·prod(extent)) —
   the assembled form needs 2.3 GB per level at the production (48,48,48,80) grid, this needs
   kilobytes plus two ping-pong scratch buffers. `inner` folds the trailing dims AND the
   channel count, so the kernel never sees the dimensionality. */
#version 460
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) readonly buffer Meta {
    int n_out; int n_in; int outer; int inner; int accum; } meta;
layout(set=0,binding=1,std430) readonly buffer Ptr { int ptr[]; };
layout(set=0,binding=2,std430) readonly buffer Col { int col[]; };
layout(set=0,binding=3,std430) readonly buffer Val { float val[]; };
layout(set=0,binding=4,std430) readonly buffer Xb  { float x[]; };
layout(set=0,binding=5,std430)          buffer Yb  { float y[]; };
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    uint total  = uint(meta.outer)*uint(meta.n_out)*uint(meta.inner);
    for(uint g=gl_GlobalInvocationID.x; g<total; g+=stride){
        int i = int(g % uint(meta.inner));
        int t = int(g / uint(meta.inner));
        int j = t % meta.n_out;
        int o = t / meta.n_out;
        float acc = 0.0;
        for(int k=ptr[j]; k<ptr[j+1]; ++k)
            acc += val[k]*x[(o*meta.n_in + col[k])*meta.inner + i];
        y[g] = (meta.accum != 0) ? y[g]+acc : acc;
    }
}
