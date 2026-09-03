/* vkflow — sparse CSR matvec for the BPX multilevel transfer (prolong P / restrict Pᵀ).
   y[row,ch] = (accum? y : 0) + Σ_{k=ptr[row]}^{ptr[row+1]} val[k]·x[col[k],ch].
   The resize is per-channel identical, so the CSR is built on the SPATIAL grid and the
   kernel sweeps the nch channels. Periodic wrap + Greville alignment are baked into col/val
   (tensor product of the validated _axis_resize_matrix factors). */
#version 460
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) readonly buffer Meta { int n_rows; int nch; int accum; } meta;
layout(set=0,binding=1,std430) readonly buffer Ptr { int ptr[]; };
layout(set=0,binding=2,std430) readonly buffer Col { int col[]; };
layout(set=0,binding=3,std430) readonly buffer Val { float val[]; };
layout(set=0,binding=4,std430) readonly buffer Xb  { float x[]; };
layout(set=0,binding=5,std430)          buffer Yb  { float y[]; };
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    uint total  = uint(meta.n_rows)*uint(meta.nch);
    for(uint o=gl_GlobalInvocationID.x; o<total; o+=stride){
        int row = int(o)/meta.nch, ch = int(o)%meta.nch;
        float acc = 0.0;
        for(int k=ptr[row]; k<ptr[row+1]; ++k) acc += val[k]*x[col[k]*meta.nch + ch];
        y[o] = (meta.accum != 0) ? y[o]+acc : acc;
    }
}
