/* vkflow — CG step length on-device: al[0]=rz/pAp (clamped), nal[0]=-al[0]. */
#version 460
layout(local_size_x = 1) in;
layout(set=0,binding=0,std430) readonly buffer RZ  { float rz[];  };
layout(set=0,binding=1,std430) readonly buffer PAP { float pap[]; };
layout(set=0,binding=2,std430)          buffer AL  { float al[];  };
layout(set=0,binding=3,std430)          buffer NAL { float nal[]; };
void main(){ float p = pap[0]; float a = (p > 1e-30) ? rz[0]/p : 0.0; al[0]=a; nal[0]=-a; }
