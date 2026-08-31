/* vkflow — max-reduction: m[0] = max(m[0], max_i x[i]).  diag(H_GN) ≥ 0, so the
   float bit-pattern is monotonic and atomicMax on uint(bits) gives the float max
   without a host round-trip of the whole buffer. Host zeroes m, reads uintBitsToFloat. */
#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) readonly buffer X { float x[]; };
layout(set=0,binding=1,std430)          buffer O { uint  m[]; };
layout(set=0,binding=2,std430) readonly buffer M { int n; } meta;
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    float acc = 0.0;
    for(uint i=gl_GlobalInvocationID.x; i<uint(meta.n); i+=stride) acc = max(acc, x[i]);
    float w = subgroupMax(acc);
    if (subgroupElect()) atomicMax(m[0], floatBitsToUint(w));
}
