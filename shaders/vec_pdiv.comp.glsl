/* vkflow — CG preconditioner solve: z = x / (d + a). */
#version 460
layout(local_size_x = 256) in;
layout(set=0,binding=3,std430) readonly buffer M { int n; float a; float b; } meta;
layout(set=0,binding=0,std430) readonly buffer X { float x[]; };
layout(set=0,binding=1,std430) readonly buffer D { float d[]; };
layout(set=0,binding=2,std430)          buffer Z { float z[]; };
void main(){
    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;
    for(uint i=gl_GlobalInvocationID.x; i<uint(meta.n); i+=stride) z[i]=x[i]/(d[i]+meta.a);
}
