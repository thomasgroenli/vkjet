/* vkflow — CG vector dot: out[0] += Σ_i a[i]·b[i] (subgroup-reduced). */
#version 460
#extension GL_EXT_shader_atomic_float : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 256) in;
layout(set=0,binding=3,std430) readonly buffer M { int n; float a; float b; } meta;
layout(set=0,binding=0,std430) readonly buffer A { float xa[]; };
layout(set=0,binding=1,std430) readonly buffer B { float xb[]; };
layout(set=0,binding=2,std430)          buffer O { float outv[]; };
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    float acc=0.0;
    for(uint i=gl_GlobalInvocationID.x; i<uint(meta.n); i+=stride) acc += xa[i]*xb[i];
    float w = subgroupAdd(acc);
    if (subgroupElect()) atomicAdd(outv[0], w);
}
