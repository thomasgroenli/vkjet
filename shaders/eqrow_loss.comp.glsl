/* vkflow — generic jet-row gradient: rows encode arbitrary (≤ quadratic)
 * residuals of the field jet via a per-file operator table.
 *
 *   r = Σ_lin eff·fld[slot][ch] + Σ_quad eff·fld[s1][c1]·fld[s2][c2] − s
 *   L += ½·scale·w·r² ;  eff = val·(cix<0 ? 1 : rowc[cix])
 *
 * Jet slots (jet2-v1): 0 value · 1..4 ∂t,∂x,∂y,∂z · 5..8 ∂tt,∂xx,∂yy,∂zz ·
 * 9..11 ∂t∂x,∂t∂y,∂t∂z · 12..14 ∂x∂y,∂x∂z,∂y∂z. World-unit derivatives
 * (dcoefs/ddcoefs carry the chain rule). Per-combo Jacobian row A[ch] is
 * built directly from the (warp-uniform) entry lists — no dense cotangent
 * array. Plain atomic scatter (correctness-first, like pde_ns5_grad).
 *
 * optab_i = [lin_off(n_ops+1) | quad_off(n_ops+1) | lin_pack | quad_pack]
 *   lin_pack  = slot<<8 | ch<<4 | (cix+1)
 *   quad_pack = s1<<20 | c1<<16 | s2<<12 | c2<<8 | (cix+1)
 * optab_f = [lin_val | quad_val]  (nnz_lin in meta splits the sections)
 *
 * Spec consts: 0 SPEC_STRIDE · 1 SPEC_TABLE_PERIOD.
 * Bindings: 0 Meta · 1 x · 2 primal · 3 coefs · 4 table · 5 dcoefs ·
 *   6 ddcoefs · 7 rowrec{op,w,s,pad} · 8 rowc(f4[N*8]) · 9 optab_i ·
 *   10 optab_f · 11 grad(float, atomic).
 */
#version 460
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_atomic_float : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define ND 4
#define NF 15
#define NCPR 8
/* NCH is declared by the ROW FILE: the operator table references channel
   indices, so the objective is not well defined without it. A
   specialization constant lets one SPIR-V module serve any channel count
   while the driver still sizes fld[NF][NCH]/A[NCH] exactly. */
layout(constant_id = 2) const int NCH = 5;
layout(constant_id = 0) const int SPEC_STRIDE = 0x7fffffff;
layout(constant_id = 1) const int SPEC_TABLE_PERIOD = 0;
layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) readonly buffer Meta {
    int ndim; int n_samples; int n_channels; int num_combos;
    float scale; int n_ops; int nnz_lin; int pad0;
    int coef_period[16], order[16], degp1[16], stride[16], table_period[16],
        primal_extent[16], pstride[16], coef_offset[16], table_offset[16];
} meta;
layout(set = 0, binding = 1, std430) readonly buffer XB { float x[]; };
layout(set = 0, binding = 2, std430) readonly buffer PB { float primal[]; };
layout(set = 0, binding = 3, std430) readonly buffer CB { float coefs[]; };
layout(set = 0, binding = 4, std430) readonly buffer TB { int table_data[]; };
layout(set = 0, binding = 5, std430) readonly buffer DB { float dcoefs[]; };
layout(set = 0, binding = 6, std430) readonly buffer QB { float ddcoefs[]; };
layout(set = 0, binding = 7, std430) readonly buffer RR { int rowrec[]; };
layout(set = 0, binding = 8, std430) readonly buffer RC { float rowc[]; };
layout(set = 0, binding = 9, std430) readonly buffer OI { int optab_i[]; };
layout(set = 0, binding = 10, std430) readonly buffer OF { float optab_f[]; };
layout(set = 0, binding = 11, std430)          buffer LB { float loss[]; };

float hv(int p,int d,float xv){ float r=coefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+coefs[p+k]; return r; }
float hd(int p,int d,float xv){ float r=dcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+dcoefs[p+k]; return r; }
float hq(int p,int d,float xv){ float r=ddcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+ddcoefs[p+k]; return r; }
int pmod(int a,int b){ int r=a%b; return r<0?r+b:r; }

/* full second-order jet tap weights (jet2-v1 slot table) */
void combo(int i, int ii[ND], float fr[ND], int pob[ND], out int prim, out float W[NF]) {
    int reduce=i; prim=0; float pv[ND], pd[ND], pq[ND];
    [[unroll]] for (int d=0; d<ND; d++) {
        int od=meta.order[d], gd=meta.degp1[d], idx=reduce%od; reduce/=od;
        int sd=(SPEC_STRIDE!=0x7fffffff)?SPEC_STRIDE:meta.stride[d];
        int tp=(SPEC_TABLE_PERIOD!=0)?SPEC_TABLE_PERIOD:meta.table_period[d];
        int tr=pmod(ii[d],tp);
        prim+=pmod(ii[d]*sd+table_data[meta.table_offset[d]+tr*od+idx], meta.primal_extent[d])*meta.pstride[d];
        int po=pob[d]+idx*gd; pv[d]=hv(po,gd-1,fr[d]); pd[d]=hd(po,gd-1,fr[d]); pq[d]=hq(po,gd-1,fr[d]);
    }
    float pre[ND+1]; pre[0]=1.0; [[unroll]] for(int d=0;d<ND;d++) pre[d+1]=pre[d]*pv[d];
    W[0]=pre[ND]; float suf=1.0;
    [[unroll]] for(int a=ND-1;a>=0;a--){
        W[1+a]=pre[a]*pd[a]*suf;
        W[5+a]=pre[a]*pq[a]*suf;
        suf*=pv[a];
    }
    W[9]=pd[0]*pd[1]*pv[2]*pv[3];  W[10]=pd[0]*pv[1]*pd[2]*pv[3];
    W[11]=pd[0]*pv[1]*pv[2]*pd[3]; W[12]=pv[0]*pd[1]*pd[2]*pv[3];
    W[13]=pv[0]*pd[1]*pv[2]*pd[3]; W[14]=pv[0]*pv[1]*pd[2]*pd[3];
}

void main() {
    uint sn = gl_GlobalInvocationID.x;
    float my = 0.0; bool ok = (int(sn) < meta.n_samples);
    int ii[ND]; float fr[ND]; int pob[ND];
    if (ok) [[unroll]] for (int d=0;d<ND;d++) {
        float xv=x[sn*uint(ND)+uint(d)]; if (isnan(xv)){ok=false;break;}
        float ip=floor(xv); ii[d]=int(ip); fr[d]=xv-ip;
        pob[d]=meta.coef_offset[d]+pmod(ii[d],meta.coef_period[d])*meta.order[d]*meta.degp1[d];
    }
    if (ok) {
        float fld[NF][NCH];
        [[unroll]] for(int j=0;j<NF;j++) for(int ch=0;ch<NCH;ch++) fld[j][ch]=0.0;
        int prim; float W[NF];
        for (int i=0;i<meta.num_combos;i++){ combo(i,ii,fr,pob,prim,W);
            [[unroll]] for(int j=0;j<NF;j++) for(int ch=0;ch<NCH;ch++) fld[j][ch]+=W[j]*primal[prim+ch]; }
        int op = rowrec[sn*4u];
        float rw = intBitsToFloat(rowrec[sn*4u+1u]);
        float rs = intBitsToFloat(rowrec[sn*4u+2u]);
        int nops1 = meta.n_ops + 1;
        int l0 = optab_i[op], l1 = optab_i[op+1];
        int q0 = optab_i[nops1+op], q1 = optab_i[nops1+op+1];
        int lbase = 2*nops1, qbase = 2*nops1 + meta.nnz_lin;
        float r = -rs;
        for (int e=l0; e<l1; e++) {
            int pk = optab_i[lbase+e]; float v = optab_f[e];
            int cixp1 = pk & 15; if (cixp1>0) v *= rowc[sn*uint(NCPR)+uint(cixp1-1)];
            int slk = (pk>>8);   /* slot NF = order-0 term */
        r += (slk < NF) ? v * fld[slk][(pk>>4)&15] : v;
        }
        for (int e=q0; e<q1; e++) {
            int pk = optab_i[qbase+e]; float v = optab_f[meta.nnz_lin+e];
            int cixp1 = pk & 255; if (cixp1>0) v *= rowc[sn*uint(NCPR)+uint(cixp1-1)];
            r += v * fld[(pk>>20)&15][(pk>>16)&15] * fld[(pk>>12)&15][(pk>>8)&15];
        }
        my = 0.5 * meta.scale * rw * r * r;
    }
    float warp = subgroupAdd(my);
    if (subgroupElect()) atomicAdd(loss[0], warp);
}
