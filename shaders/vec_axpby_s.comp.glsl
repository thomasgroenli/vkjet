/* vkflow — CG axpby with DEVICE-resident scalars: z = A[0]·x + B[0]·y.  The
   coefficients come from on-device 1-float buffers (written by cg_alpha/cg_beta),
   so α/β never round-trip through the host. z may alias y. */
#version 460
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) readonly buffer X { float x[]; };
layout(set=0,binding=1,std430) readonly buffer Y { float y[]; };
layout(set=0,binding=2,std430)          buffer Z { float z[]; };
layout(set=0,binding=3,std430) readonly buffer A { float a[]; };
layout(set=0,binding=4,std430) readonly buffer B { float b[]; };
layout(set=0,binding=5,std430) readonly buffer M { int n; } meta;
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    float aa = a[0], bb = b[0];
    for(uint i=gl_GlobalInvocationID.x; i<uint(meta.n); i+=stride) z[i] = aa*x[i] + bb*y[i];
}
