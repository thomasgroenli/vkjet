/* vkflow — CG direction update on-device: be[0]=rzn/rz, then advance rz[0]=rzn. */
#version 460
layout(local_size_x = 1) in;
layout(set=0,binding=0,std430) readonly buffer RZN { float rzn[]; };
layout(set=0,binding=1,std430)          buffer RZ  { float rz[];  };
layout(set=0,binding=2,std430)          buffer BE  { float be[];  };
void main(){ float r = rz[0]; float n = rzn[0]; be[0] = (r > 1e-30) ? n/r : 0.0; rz[0]=n; }
