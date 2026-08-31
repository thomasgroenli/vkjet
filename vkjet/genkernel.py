"""JIT-generated fused kernels for jet-row operators (structure-hash cache).

The full fused permutation space over the jet is 2^2925 structures —
unenumerable. But any SINGLE operator's structure (which jet sites its
L/Q entries touch) can be compiled into a specialized kernel on demand:
the eqrow entry loops unroll into literal FMAs, and the gather touches only
the used (slot, channel) sites instead of the full 15×5 jet. Coefficient
VALUES stay in a small buffer read at literal indices, so one cached shader
serves every value assignment (any ν, any payload) of the same structure.

Cache: ~/.cache/vkjet/genspv/<srchash>_<kind>.spv, keyed by a hash of the
EMITTED GLSL itself (plus GEN_VERSION). Keying on the source rather than on
a hand-maintained projection of the operator makes key ≡ code by
construction: two operators share a cache entry only when their shaders are
byte-identical, and any edit to the emitters, the templates, or the
constants they bake in (NCPR, ND, MIXED_PAIRS, the binding layout)
invalidates the cache automatically. structure_hash remains the operator's
structural identity for introspection. Entries are compiled to a temp file
and os.replace()d into position, and a cache hit is accepted only if the
file is a well-formed SPIR-V module, so an interrupted or concurrent
compile can never be mistaken for a valid one.
Compilation needs glslc or a modern glslangValidator ($GLSLC / $GLSLANG
honoured, same as shaders/build.sh); absent toolchain → caller falls back
to the generic EqRowTerm. Every generated term is PARITY-VERIFIED against
the generic kernel on a random micro-batch before first use — all FOUR
kernels (loss, grad, diag, hvp), over the full encoded domain including the
boundary cells — so a mislabeled or miscompiled kernel can never silently
corrupt the objective or the Gauss-Newton operator built from it.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

import numpy as np

from .context import STORAGE
from .eqrow import (EqRowTerm, OperatorTable, NF, NCH, NCPR, MIXED_PAIRS,
                    pack_eqrow_meta)

GEN_VERSION = "2"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vkjet", "genspv")
SPIRV_MAGIC = b"\x03\x02\x23\x07"     # 0x07230203, little-endian on disk

KINDS = ("grad", "loss", "diag", "hvp")


def find_compiler():
    """→ (kind, path) with kind in {"glslc", "glslang"}, or None.

    Discovery order matches shaders/build.sh: $GLSLC, $GLSLANG, then PATH,
    then the micromamba py314 env as a last resort."""
    from shutil import which
    env_c, env_g = os.environ.get("GLSLC"), os.environ.get("GLSLANG")
    if env_c:
        return ("glslc", env_c)
    if env_g:
        return ("glslang", env_g)
    p = which("glslc")
    if p:
        return ("glslc", p)
    for cand in (which("glslangValidator"),
                 os.path.expanduser("~/micromamba/envs/py314/bin/glslangValidator")):
        if cand and os.path.exists(cand):
            return ("glslang", cand)
    return None


# entry orderings — MUST equal the keys structure_hash canonicalizes on, so
# that the emitted code (and the vals[] packing that mirrors it) is a
# function of the hashed structure alone and never of the coefficients.
def _lin_key(e):
    return (e[0], e[1], e[3])


def _quad_key(q):
    return (q[0], q[1], q[2], q[3], q[5])


def source_key(srcs):
    """Cache identity of a generated kernel SET: a hash of the GLSL itself."""
    h = hashlib.sha256(GEN_VERSION.encode())
    for kind in KINDS:
        h.update(b"\x00" + kind.encode() + b"\x00" + srcs[kind].encode())
    return h.hexdigest()[:20]


def valid_spv(path):
    """True if `path` is a plausible SPIR-V module (present, word-sized,
    correct magic). Guards against a partial file left by an interrupted or
    concurrent compile — os.path.exists alone accepts those forever."""
    try:
        sz = os.path.getsize(path)
        if sz < 20 or sz % 4:
            return False
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic in (SPIRV_MAGIC, SPIRV_MAGIC[::-1])
    except OSError:
        return False


def structure_hash(ops: OperatorTable, k: int) -> str:
    """Indices-only identity of operator k's structure (values stripped).

    NOT the cache key — see source_key(). Kept as the operator's structural
    identity, and as the ordering the emitters canonicalize on."""
    lin = sorted(_lin_key(e) for e in ops.lin[k])
    quad = sorted(_quad_key(q) for q in ops.quad[k])
    key = f"jet2v1/nch{NCH}/gen{GEN_VERSION}/{lin}/{quad}".encode()
    return hashlib.sha256(key).hexdigest()[:20]


# --------------------------------------------------------------------------- #
# GLSL emission                                                               #
# --------------------------------------------------------------------------- #
_HEAD = """#version 460
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_atomic_float : require
{extra_ext}
#define ND 4
#define NCPR {ncpr}
layout(constant_id = 0) const int SPEC_STRIDE = 0x7fffffff;
layout(constant_id = 1) const int SPEC_TABLE_PERIOD = 0;
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Meta {{
    int ndim; int n_samples; int n_channels; int num_combos;
    float scale; int n_ops; int nnz_lin; int pad0;
    int coef_period[16], order[16], degp1[16], stride[16], table_period[16],
        primal_extent[16], pstride[16], coef_offset[16], table_offset[16];
}} meta;
layout(set = 0, binding = 1, std430) readonly buffer XB {{ float x[]; }};
layout(set = 0, binding = 2, std430) readonly buffer PB {{ float primal[]; }};
layout(set = 0, binding = 3, std430) readonly buffer CB {{ float coefs[]; }};
layout(set = 0, binding = 4, std430) readonly buffer TB {{ int table_data[]; }};
layout(set = 0, binding = 5, std430) readonly buffer DB {{ float dcoefs[]; }};
layout(set = 0, binding = 6, std430) readonly buffer QB {{ float ddcoefs[]; }};
layout(set = 0, binding = 7, std430) readonly buffer RR {{ int rowrec[]; }};
layout(set = 0, binding = 8, std430) readonly buffer RC {{ float rowc[]; }};
layout(set = 0, binding = 9, std430) readonly buffer VV {{ float vals[]; }};
{out_bindings}

float hv(int p,int d,float xv){{ float r=coefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+coefs[p+k]; return r; }}
float hd(int p,int d,float xv){{ float r=dcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+dcoefs[p+k]; return r; }}
float hq(int p,int d,float xv){{ float r=ddcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+ddcoefs[p+k]; return r; }}
int pmod(int a,int b){{ int r=a%b; return r<0?r+b:r; }}

void combo(int i, int ii[ND], float fr[ND], int pob[ND], out int prim,
{w_out_decl}) {{
    int reduce=i; prim=0; float pv[ND];{pd_decl}{pq_decl}
    [[unroll]] for (int d=0; d<ND; d++) {{
        int od=meta.order[d], gd=meta.degp1[d], idx=reduce%od; reduce/=od;
        int sd=(SPEC_STRIDE!=0x7fffffff)?SPEC_STRIDE:meta.stride[d];
        int tp=(SPEC_TABLE_PERIOD!=0)?SPEC_TABLE_PERIOD:meta.table_period[d];
        int tr=pmod(ii[d],tp);
        prim+=pmod(ii[d]*sd+table_data[meta.table_offset[d]+tr*od+idx], meta.primal_extent[d])*meta.pstride[d];
        int po=pob[d]+idx*gd; pv[d]=hv(po,gd-1,fr[d]);{pd_calc}{pq_calc}
    }}
{w_exprs}
}}
"""

_PRELUDE = """    int ii[ND]; float fr[ND]; int pob[ND];
    [[unroll]] for (int d=0;d<ND;d++) {{
        float xv=x[sn*uint(ND)+uint(d)]; if (isnan(xv)) {nanact}
        float ip=floor(xv); ii[d]=int(ip); fr[d]=xv-ip;
        pob[d]=meta.coef_offset[d]+pmod(ii[d],meta.coef_period[d])*meta.order[d]*meta.degp1[d];
    }}
"""


def _slot_expr(slot):
    """W expression for a jet slot from pv/pd/pq factors."""
    if slot == 0:
        f = ["pv[0]", "pv[1]", "pv[2]", "pv[3]"]
    elif 1 <= slot <= 4:
        a = slot - 1
        f = [f"pd[{d}]" if d == a else f"pv[{d}]" for d in range(4)]
    elif 5 <= slot <= 8:
        a = slot - 5
        f = [f"pq[{d}]" if d == a else f"pv[{d}]" for d in range(4)]
    else:
        a, b = MIXED_PAIRS[slot - 9]
        f = [f"pd[{d}]" if d in (a, b) else f"pv[{d}]" for d in range(4)]
    return "*".join(f)


def emit_shaders(ops: OperatorTable, k: int):
    """→ {"grad": src, "loss": src, "diag": src, "hvp": src} for operator k,
    plus the value-packing order (sorted entries)."""
    lin = sorted(ops.lin[k], key=_lin_key)
    quad = sorted(ops.quad[k], key=_quad_key)
    if not lin and not quad:
        raise ValueError(
            f"operator {ops.names[k]!r} has no entries — nothing to specialize "
            f"(residual is just -s; use the generic EqRowTerm)")
    used_slots = sorted({e[0] for e in lin}
                        | {q[0] for q in quad} | {q[2] for q in quad})
    sites = sorted({(e[0], e[1]) for e in lin}
                   | {(q[0], q[1]) for q in quad} | {(q[2], q[3]) for q in quad})
    chans = sorted({e[1] for e in lin} | {q[1] for q in quad}
                   | {q[3] for q in quad})
    need_pd = any(1 <= s <= 4 or s >= 9 for s in used_slots)
    need_pq = any(5 <= s <= 8 for s in used_slots)

    head_fmt = dict(
        ncpr=NCPR,
        pd_decl=" float pd[ND];" if need_pd else "",
        pq_decl=" float pq[ND];" if need_pq else "",
        pd_calc=" pd[d]=hd(po,gd-1,fr[d]);" if need_pd else "",
        pq_calc=" pq[d]=hq(po,gd-1,fr[d]);" if need_pq else "",
        w_out_decl=", ".join(f"out float W{s}" for s in used_slots),
        w_exprs="\n".join(f"    W{s} = {_slot_expr(s)};" for s in used_slots))

    wargs = ", ".join(f"W{s}" for s in used_slots)

    # per-row constants: unrolled effective values v<e>
    vlines = []
    for e, (sl, ch, val, cix) in enumerate(lin):
        m = f"*rowc[sn*uint(NCPR)+{cix}u]" if cix >= 0 else ""
        vlines.append(f"    float v{e} = vals[{e}]{m};")
    for j, (s1, c1, s2, c2, val, cix) in enumerate(quad):
        e = len(lin) + j
        m = f"*rowc[sn*uint(NCPR)+{cix}u]" if cix >= 0 else ""
        vlines.append(f"    float v{e} = vals[{e}]{m};")
    vblock = "\n".join(vlines)

    # gather: scalar accumulators for used sites only
    fdecl = "\n".join(f"    float f{s}_{ch} = 0.0;" for s, ch in sites)
    facc = "\n".join(f"        f{s}_{ch} += W{s}*primal[prim+{ch}];"
                     for s, ch in sites)
    wdecl = "    float " + ", ".join(f"W{s}" for s in used_slots) + ";"
    gather = (f"{fdecl}\n{wdecl}\n    int prim;\n"
              f"    for (int i=0;i<meta.num_combos;i++){{\n"
              f"        combo(i,ii,fr,pob,prim,{wargs});\n{facc}\n    }}")

    # residual
    rterms = [f"v{e}*f{sl}_{ch}" for e, (sl, ch, _, _) in enumerate(lin)]
    rterms += [f"v{len(lin)+j}*f{s1}_{c1}*f{s2}_{c2}"
               for j, (s1, c1, s2, c2, _, _) in enumerate(quad)]
    rexpr = " + ".join(rterms) if rterms else "0.0"

    # per-combo Jacobian row A_c for used channels
    aterms = {c: [] for c in chans}
    for e, (sl, ch, _, _) in enumerate(lin):
        aterms[ch].append(f"v{e}*W{sl}")
    for j, (s1, c1, s2, c2, _, _) in enumerate(quad):
        e = len(lin) + j
        aterms[c1].append(f"v{e}*f{s2}_{c2}*W{s1}")
        aterms[c2].append(f"v{e}*f{s1}_{c1}*W{s2}")
    ablock = "\n".join(f"        float A{c} = " + " + ".join(aterms[c]) + ";"
                       for c in chans)

    def scatter(body):
        return (f"    for (int i=0;i<meta.num_combos;i++){{\n"
                f"        combo(i,ii,fr,pob,prim,{wargs});\n"
                f"{ablock}\n{body}\n    }}")

    rw = ("    float rw = intBitsToFloat(rowrec[sn*4u+1u]);\n"
          "    float rs = intBitsToFloat(rowrec[sn*4u+2u]);")

    grad_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{_PRELUDE.format(nanact="return;")}{rw}
{vblock}
{gather}
    float r = -rs + {rexpr};
    float a = meta.scale * rw * r;
{scatter(chr(10).join(f"        atomicAdd(grad[prim+{c}], a*A{c});" for c in chans))}
}}
"""
    loss_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    float my = 0.0; bool ok = (int(sn) < meta.n_samples);
    int ii[ND]; float fr[ND]; int pob[ND];
    if (ok) [[unroll]] for (int d=0;d<ND;d++) {{
        float xv=x[sn*uint(ND)+uint(d)]; if (isnan(xv)){{ok=false;break;}}
        float ip=floor(xv); ii[d]=int(ip); fr[d]=xv-ip;
        pob[d]=meta.coef_offset[d]+pmod(ii[d],meta.coef_period[d])*meta.order[d]*meta.degp1[d];
    }}
    if (ok) {{
{rw}
{vblock}
{gather}
        float r = -rs + {rexpr};
        my = 0.5 * meta.scale * rw * r * r;
    }}
    float warp = subgroupAdd(my);
    if (subgroupElect()) atomicAdd(loss[0], warp);
}}
"""
    diag_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{_PRELUDE.format(nanact="return;")}{rw}
{vblock}
{gather}
    float sw = meta.scale * rw;
{scatter(chr(10).join(f"        atomicAdd(diag[prim+{c}], sw*A{c}*A{c});" for c in chans))}
}}
"""
    jv = " + ".join(f"A{c}*vvec[prim+{c}]" for c in chans)
    hvp_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{_PRELUDE.format(nanact="return;")}{rw}
{vblock}
{gather}
    float Jv = 0.0;
{scatter(f"        Jv += {jv};")}
    float y = meta.scale * rw * Jv;
{scatter(chr(10).join(f"        atomicAdd(outv[prim+{c}], A{c}*y);" for c in chans))}
}}
"""
    outs = {
        "grad": ("", 'layout(set = 0, binding = 10, std430)          buffer GB { float grad[]; };'),
        "loss": ("#extension GL_KHR_shader_subgroup_basic : require\n"
                 "#extension GL_KHR_shader_subgroup_arithmetic : require",
                 'layout(set = 0, binding = 10, std430)          buffer LB { float loss[]; };'),
        "diag": ("", 'layout(set = 0, binding = 10, std430)          buffer DGB { float diag[]; };'),
        "hvp": ("", 'layout(set = 0, binding = 10, std430) readonly buffer VB { float vvec[]; };\n'
                'layout(set = 0, binding = 11, std430)          buffer OB { float outv[]; };'),
    }
    mains = {"grad": grad_main, "loss": loss_main, "diag": diag_main,
             "hvp": hvp_main}
    srcs = {}
    for kind, (ext, ob) in outs.items():
        srcs[kind] = (_HEAD.format(extra_ext=ext, out_bindings=ob, **head_fmt)
                      + mains[kind])
    values = np.asarray([e[2] for e in lin] + [q[4] for q in quad], np.float32)
    return srcs, values


def _compile(src, out_path, compiler):
    """Compile `src` to `out_path` ATOMICALLY: the compiler writes a unique
    temp file in the same directory and it is os.replace()d into position, so
    a reader never sees a partially written module and two racing writers both
    end up with a complete one. On failure the offending GLSL is kept next to
    the cache entry (<out_path>.failed.comp) — glslang's line numbers are
    useless without it."""
    kind, path = compiler
    out_dir = os.path.dirname(out_path) or "."
    with tempfile.NamedTemporaryFile("w", suffix=".comp", delete=False) as f:
        f.write(src)
        tmp = f.name
    fd, tmp_spv = tempfile.mkstemp(dir=out_dir, suffix=".spv.tmp")
    os.close(fd)
    try:
        if kind == "glslc":
            cmd = [path, "-fshader-stage=comp", "--target-env=vulkan1.1",
                   "-O", tmp, "-o", tmp_spv]
        else:
            cmd = [path, "--target-env", "vulkan1.1", "-S", "comp",
                   tmp, "-o", tmp_spv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not valid_spv(tmp_spv):
            kept = out_path + ".failed.comp"
            try:
                os.replace(tmp, kept)
                tmp = None
            except OSError:
                kept = "(not retained)"
            raise RuntimeError(
                f"shader compile failed (rc={r.returncode}), source kept at "
                f"{kept}: {(r.stderr or r.stdout)[:2000]}")
        os.replace(tmp_spv, out_path)
        tmp_spv = None
    finally:
        if tmp is not None:
            os.unlink(tmp)
        if tmp_spv is not None:
            try:
                os.unlink(tmp_spv)
            except OSError:
                pass


class GeneratedRowTerm(EqRowTerm):
    """EqRowTerm specialized to ONE operator via a JIT-compiled shader set.
    Same 4-method protocol and binding layout family; entry loops unrolled,
    only the touched jet sites gathered; values live in a compact buffer so
    the cached shader serves every coefficient assignment of the structure."""

    def __init__(self, ctx, bases, inv_widths, ops, k, compiler=None,
                 cache_dir=None):
        compiler = compiler or find_compiler()
        if compiler is None:
            raise RuntimeError("no glslc/glslangValidator available")
        cache_dir = cache_dir or CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        srcs, values = emit_shaders(ops, k)
        key = source_key(srcs)           # key ≡ code: the vals[] packing below
        spv = {kind: os.path.join(cache_dir, f"{key}_{kind}.spv")
               for kind in KINDS}        # is emitted by this same call
        for kind, p in spv.items():
            if not valid_spv(p):         # absent, partial, or not SPIR-V
                _compile(srcs[kind], p, compiler)
        # EqRowTerm init minus optab: replicate the buffer setup
        sub = OperatorTable()
        sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        super().__init__(ctx, bases, inv_widths, sub)
        self.grad_program = ctx.program(spv["grad"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.loss_program = ctx.program(spv["loss"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.diag_program = ctx.program(spv["diag"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.hvp_program = ctx.program(spv["hvp"], bindings=[STORAGE] * 12,
                                       spec_constant_ids=[0, 1])
        self.vals_buf = ctx.buffer(max(values.nbytes, 4))
        self.vals_buf.upload(values)
        self.structure = structure_hash(ops, k)
        self.cache_key = key

    def _shared(self, coef_buf):
        b = self._batch
        return [b["mb"], b["xb"], coef_buf, self.coefs_buf, self.table_buf,
                self.dcoefs_buf, self.ddcoefs_buf, b["rrb"], b["rcb"],
                self.vals_buf]

    def bind_rows(self, axes, x, w, s, c, scale=1.0):
        """Bind rows of THIS operator (op column implicit)."""
        return self.bind_batch(axes.encode(x), np.zeros(len(x), np.int32),
                               w, s, c, scale=scale)


# --------------------------------------------------------------------------- #
# Parity verification against the generic referee                             #
# --------------------------------------------------------------------------- #
_VERIFIED = set()          # (cache_key, basis shape) already checked this run


def _basis_shape(bases):
    """What the emitted/compiled kernel actually depends on: the unrolled combo
    structure (order), the Horner degree (degp1) and the spec-constant inputs
    (stride, table_period). coef_period / primal_extent are deliberately absent
    — they are read from meta at runtime, identically by the generated and the
    generic kernel, so they cannot change whether the two agree. Excluding them
    lets a dyadic ladder verify ONCE, at the coarsest grid, instead of paying
    an O(coefficient count) check at every stage."""
    return tuple((b.order, b.degp1, b.stride, b.table_period) for b in bases)


def verify_points(bases, rng, n):
    """n random ENCODED points covering the full [0, primal_extent) range that
    Axes.encode produces, with the first and last cell of every dimension
    guaranteed to be hit (a boundary-only defect used to slip through)."""
    g = np.asarray([b.primal_extent for b in bases], np.float64)
    x = rng.uniform(0, 1, (n, 4)) * g
    i = 0
    for d in range(len(g)):
        for first in (True, False):
            if i >= n:
                break
            x[i, d] = rng.uniform(0, 1) if first else g[d] - rng.uniform(0, 1)
            i += 1
    return np.minimum(x, g - 1e-4).astype(np.float32)


def _compare(ref, term, ctx, nco, rtol, what):
    """Run all FOUR kernels on both terms and compare. Both must already be
    bound to the same rows."""
    rng = np.random.default_rng(12345)
    cb = ctx.buffer(nco * 4)
    cb.upload((rng.standard_normal(nco) * 0.3).astype(np.float32))
    vb = ctx.buffer(nco * 4)
    vb.upload((rng.standard_normal(nco) * 0.3).astype(np.float32))
    lb = ctx.buffer(4); ab = ctx.buffer(nco * 4)
    outs = []
    for t in (ref, term):
        lb.zero(); t.loss(cb, lb)
        loss = float(lb.download(np.float32, 1)[0])
        res = [loss]
        for run in (lambda: t.accumulate(cb, ab),
                    lambda: t.accumulate_diag(cb, ab),
                    lambda: t.hvp(cb, vb, ab)):
            ab.zero(); run(); res.append(ab.download(np.float32, nco))
        outs.append(res)
    a, b = outs
    if abs(b[0] - a[0]) > rtol * max(abs(a[0]), 1e-9):
        raise RuntimeError(f"{what} loss parity failed: {a[0]} vs {b[0]}")
    for name, x0, x1 in zip(("grad", "diag", "hvp"), a[1:], b[1:]):
        rel = np.linalg.norm(x1 - x0) / max(np.linalg.norm(x0), 1e-9)
        if rel > rtol:
            raise RuntimeError(f"{what} {name} parity failed: rel {rel}")
    return True


def verify_generated(ctx, term, bases, inv_widths, ops, k, seed=0, n=64,
                     rtol=2e-4, force=False):
    """Parity-check a generated term against the generic EqRowTerm on a random
    micro-batch — loss, grad, diag AND hvp. Raises on mismatch.

    Memoized on (term.cache_key, basis shape): in a dyadic ladder the check
    runs once, at the coarsest grid, instead of once per stage (the cost is
    O(coefficient count), not O(batch)). Pass force=True to re-check.
    Leaves `term` bound as the caller left it."""
    memo = (getattr(term, "cache_key", None), _basis_shape(bases))
    if not force and memo in _VERIFIED:
        return True
    rng = np.random.default_rng(seed)
    nco = int(np.prod([b.primal_extent for b in bases])) * NCH
    x_enc = verify_points(bases, rng, n)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    s = rng.standard_normal(n).astype(np.float32) * 0.1
    c = rng.standard_normal((n, NCPR)).astype(np.float32)

    sub = OperatorTable()
    sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
    ref = EqRowTerm(ctx, bases, inv_widths, sub)
    ref.bind_batch(x_enc, np.zeros(n, np.int32), w, s, c)
    saved = term._batch
    term.bind_batch(x_enc, np.zeros(n, np.int32), w, s, c)
    try:
        _compare(ref, term, ctx, nco, rtol, "generated-kernel")
    finally:
        term._batch = saved
    _VERIFIED.add(memo)
    return True


# --------------------------------------------------------------------------- #
# Grouped generation: one kernel per co-located operator SET (shared gather)  #
# --------------------------------------------------------------------------- #
def group_structure_hash(ops: OperatorTable, ids) -> str:
    parts = [structure_hash(ops, k) for k in ids]
    key = ("group/" + "/".join(parts)).encode()
    return hashlib.sha256(key).hexdigest()[:20]


_GHEAD = """#version 460
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_atomic_float : require
{extra_ext}
#define ND 4
layout(constant_id = 0) const int SPEC_STRIDE = 0x7fffffff;
layout(constant_id = 1) const int SPEC_TABLE_PERIOD = 0;
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Meta {{
    int ndim; int n_samples; int n_channels; int num_combos;
    float scale; int n_ops; int nnz_lin; int pad0;
    int coef_period[16], order[16], degp1[16], stride[16], table_period[16],
        primal_extent[16], pstride[16], coef_offset[16], table_offset[16];
}} meta;
layout(set = 0, binding = 1, std430) readonly buffer XB {{ float x[]; }};
layout(set = 0, binding = 2, std430) readonly buffer PB {{ float primal[]; }};
layout(set = 0, binding = 3, std430) readonly buffer CB {{ float coefs[]; }};
layout(set = 0, binding = 4, std430) readonly buffer TB {{ int table_data[]; }};
layout(set = 0, binding = 5, std430) readonly buffer DB {{ float dcoefs[]; }};
layout(set = 0, binding = 6, std430) readonly buffer QB {{ float ddcoefs[]; }};
layout(set = 0, binding = 7, std430) readonly buffer WB {{ float wrow[]; }};
layout(set = 0, binding = 8, std430) readonly buffer SB {{ float srow[]; }};
layout(set = 0, binding = 9, std430) readonly buffer VV {{ float vals[]; }};
{out_bindings}

float hv(int p,int d,float xv){{ float r=coefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+coefs[p+k]; return r; }}
float hd(int p,int d,float xv){{ float r=dcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+dcoefs[p+k]; return r; }}
float hq(int p,int d,float xv){{ float r=ddcoefs[p+d]; for(int k=d-1;k>=0;k--) r=r*xv+ddcoefs[p+k]; return r; }}
int pmod(int a,int b){{ int r=a%b; return r<0?r+b:r; }}

void combo(int i, int ii[ND], float fr[ND], int pob[ND], out int prim,
{w_out_decl}) {{
    int reduce=i; prim=0; float pv[ND];{pd_decl}{pq_decl}
    [[unroll]] for (int d=0; d<ND; d++) {{
        int od=meta.order[d], gd=meta.degp1[d], idx=reduce%od; reduce/=od;
        int sd=(SPEC_STRIDE!=0x7fffffff)?SPEC_STRIDE:meta.stride[d];
        int tp=(SPEC_TABLE_PERIOD!=0)?SPEC_TABLE_PERIOD:meta.table_period[d];
        int tr=pmod(ii[d],tp);
        prim+=pmod(ii[d]*sd+table_data[meta.table_offset[d]+tr*od+idx], meta.primal_extent[d])*meta.pstride[d];
        int po=pob[d]+idx*gd; pv[d]=hv(po,gd-1,fr[d]);{pd_calc}{pq_calc}
    }}
{w_exprs}
}}
"""


def emit_group_shaders(ops: OperatorTable, ids):
    """One shared-gather kernel set for co-located operators `ids` (payload
    refs disallowed). Per point: gather the UNION of touched sites once,
    evaluate every residual, weight each by its own row weight."""
    ids = list(ids)
    m = len(ids)
    if not m:
        raise ValueError("grouped generation needs at least one operator")
    for k in ids:                        # raise, not assert: under python -O
        for e in ops.lin[k]:             # a stripped assert would emit a
            if e[3] >= 0:                # kernel that silently drops c[cix]
                raise ValueError(f"grouped ops must be payload-free: "
                                 f"{ops.names[k]!r} lin entry references "
                                 f"rowc[{e[3]}]")
        for q in ops.quad[k]:
            if q[5] >= 0:
                raise ValueError(f"grouped ops must be payload-free: "
                                 f"{ops.names[k]!r} quad entry references "
                                 f"rowc[{q[5]}]")
    lin = {j: sorted(ops.lin[k], key=_lin_key) for j, k in enumerate(ids)}
    quad = {j: sorted(ops.quad[k], key=_quad_key) for j, k in enumerate(ids)}
    used_slots = sorted({e[0] for L in lin.values() for e in L}
                        | {q[0] for Q in quad.values() for q in Q}
                        | {q[2] for Q in quad.values() for q in Q})
    sites = sorted({(e[0], e[1]) for L in lin.values() for e in L}
                   | {(q[0], q[1]) for Q in quad.values() for q in Q}
                   | {(q[2], q[3]) for Q in quad.values() for q in Q})
    if not used_slots:
        raise ValueError("grouped generation needs at least one operator with "
                         "entries — every member is empty")
    need_pd = any(1 <= s <= 4 or s >= 9 for s in used_slots)
    need_pq = any(5 <= s <= 8 for s in used_slots)
    head_fmt = dict(
        pd_decl=" float pd[ND];" if need_pd else "",
        pq_decl=" float pq[ND];" if need_pq else "",
        pd_calc=" pd[d]=hd(po,gd-1,fr[d]);" if need_pd else "",
        pq_calc=" pq[d]=hq(po,gd-1,fr[d]);" if need_pq else "",
        w_out_decl=", ".join(f"out float W{s}" for s in used_slots),
        w_exprs="\n".join(f"    W{s} = {_slot_expr(s)};" for s in used_slots))
    wargs = ", ".join(f"W{s}" for s in used_slots)

    # value packing: [op0 lin | op0 quad | op1 lin | ...]
    values, vidx = [], {}
    for j in range(m):
        for e, ent in enumerate(lin[j]):
            vidx[("l", j, e)] = len(values); values.append(ent[2])
        for e, ent in enumerate(quad[j]):
            vidx[("q", j, e)] = len(values); values.append(ent[4])
    vblock = "\n".join(f"    float v{i} = vals[{i}];"
                       for i in range(len(values)))

    ws = "\n".join(f"    float w{j} = wrow[sn*{m}u+{j}u];\n"
                   f"    float s{j} = srow[sn*{m}u+{j}u];" for j in range(m))
    fdecl = "\n".join(f"    float f{s}_{c} = 0.0;" for s, c in sites)
    wdecl = "    float " + ", ".join(f"W{s}" for s in used_slots) + ";"
    facc = "\n".join(f"        f{s}_{c} += W{s}*primal[prim+{c}];"
                     for s, c in sites)
    gather = (f"{fdecl}\n{wdecl}\n    int prim;\n"
              f"    for (int i=0;i<meta.num_combos;i++){{\n"
              f"        combo(i,ii,fr,pob,prim,{wargs});\n{facc}\n    }}")

    def rexpr(j):
        t = [f"v{vidx[('l', j, e)]}*f{sl}_{ch}"
             for e, (sl, ch, _, _) in enumerate(lin[j])]
        t += [f"v{vidx[('q', j, e)]}*f{s1}_{c1}*f{s2}_{c2}"
              for e, (s1, c1, s2, c2, _, _) in enumerate(quad[j])]
        return " + ".join(t) if t else "0.0"
    rblock = "\n".join(f"    float r{j} = -s{j} + {rexpr(j)};\n"
                       f"    float a{j} = meta.scale*w{j}*r{j};"
                       for j in range(m))

    # cotangents g_site = sum_j a_j * dr_j/df_site  (for grad scatter)
    gterms = {st: [] for st in sites}
    for j in range(m):
        for e, (sl, ch, _, _) in enumerate(lin[j]):
            gterms[(sl, ch)].append(f"a{j}*v{vidx[('l', j, e)]}")
        for e, (s1, c1, s2, c2, _, _) in enumerate(quad[j]):
            v = f"v{vidx[('q', j, e)]}"
            gterms[(s1, c1)].append(f"a{j}*{v}*f{s2}_{c2}")
            gterms[(s2, c2)].append(f"a{j}*{v}*f{s1}_{c1}")
    gblock = "\n".join(f"    float g{s}_{c} = " + " + ".join(t) + ";"
                       for (s, c), t in gterms.items() if t)
    chans = sorted({c for _, c in sites})
    acc = {c: [f"W{s}*g{s}_{c}" for (s, cc) in sites if cc == c and
               gterms[(s, cc)]] for c in chans}
    grad_scatter = "\n".join(
        f"        atomicAdd(grad[prim+{c}], " + " + ".join(acc[c]) + ");"
        for c in chans if acc[c])

    # per-combo Jacobian rows A<j>_<c> (diag/hvp)
    ach = {}
    for j in range(m):
        t = {}
        for e, (sl, ch, _, _) in enumerate(lin[j]):
            t.setdefault(ch, []).append(f"v{vidx[('l', j, e)]}*W{sl}")
        for e, (s1, c1, s2, c2, _, _) in enumerate(quad[j]):
            v = f"v{vidx[('q', j, e)]}"
            t.setdefault(c1, []).append(f"{v}*f{s2}_{c2}*W{s1}")
            t.setdefault(c2, []).append(f"{v}*f{s1}_{c1}*W{s2}")
        ach[j] = t
    ablock = "\n".join(
        f"        float A{j}_{c} = " + " + ".join(ts) + ";"
        for j in range(m) for c, ts in sorted(ach[j].items()))

    def loop(body):
        return (f"    for (int i=0;i<meta.num_combos;i++){{\n"
                f"        combo(i,ii,fr,pob,prim,{wargs});\n{ablock}\n"
                f"{body}\n    }}")

    diag_body = "\n".join(
        f"        atomicAdd(diag[prim+{c}], meta.scale*(" + " + ".join(
            f"w{j}*A{j}_{c}*A{j}_{c}" for j in range(m) if c in ach[j])
        + "));" for c in chans)
    jv_body = "\n".join(                 # an entry-free member contributes
        f"        Jv{j} += " + (" + ".join(   # nothing (was: "Jv{j} += ;")
            f"A{j}_{c}*vvec[prim+{c}]" for c in sorted(ach[j])) or "0.0") + ";"
        for j in range(m))
    out_body = "\n".join(
        f"        atomicAdd(outv[prim+{c}], " + " + ".join(
            f"A{j}_{c}*y{j}" for j in range(m) if c in ach[j]) + ");"
        for c in chans)

    pre = _PRELUDE.format(nanact="return;")
    grad_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{pre}{ws}
{vblock}
{gather}
{rblock}
{gblock}
    for (int i=0;i<meta.num_combos;i++){{
        combo(i,ii,fr,pob,prim,{wargs});
{grad_scatter}
    }}
}}
"""
    loss_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    float my = 0.0; bool ok = (int(sn) < meta.n_samples);
    int ii[ND]; float fr[ND]; int pob[ND];
    if (ok) [[unroll]] for (int d=0;d<ND;d++) {{
        float xv=x[sn*uint(ND)+uint(d)]; if (isnan(xv)){{ok=false;break;}}
        float ip=floor(xv); ii[d]=int(ip); fr[d]=xv-ip;
        pob[d]=meta.coef_offset[d]+pmod(ii[d],meta.coef_period[d])*meta.order[d]*meta.degp1[d];
    }}
    if (ok) {{
{ws}
{vblock}
{gather}
{rblock}
        my = {" + ".join(f"0.5*meta.scale*w{j}*r{j}*r{j}" for j in range(m))};
    }}
    float warp = subgroupAdd(my);
    if (subgroupElect()) atomicAdd(loss[0], warp);
}}
"""
    diag_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{pre}{ws}
{vblock}
{gather}
{loop(diag_body)}
}}
"""
    jvdecl = "    float " + ", ".join(f"Jv{j} = 0.0" for j in range(m)) + ";"
    ydecl = "\n".join(f"    float y{j} = meta.scale*w{j}*Jv{j};"
                      for j in range(m))
    hvp_main = f"""void main() {{
    uint sn = gl_GlobalInvocationID.x;
    if (int(sn) >= meta.n_samples) return;
{pre}{ws}
{vblock}
{gather}
{jvdecl}
{loop(jv_body)}
{ydecl}
{loop(out_body)}
}}
"""
    outs = {
        "grad": ("", 'layout(set = 0, binding = 10, std430)          buffer GB { float grad[]; };'),
        "loss": ("#extension GL_KHR_shader_subgroup_basic : require\n"
                 "#extension GL_KHR_shader_subgroup_arithmetic : require",
                 'layout(set = 0, binding = 10, std430)          buffer LB { float loss[]; };'),
        "diag": ("", 'layout(set = 0, binding = 10, std430)          buffer DGB { float diag[]; };'),
        "hvp": ("", 'layout(set = 0, binding = 10, std430) readonly buffer VB { float vvec[]; };\n'
                'layout(set = 0, binding = 11, std430)          buffer OB { float outv[]; };'),
    }
    mains = {"grad": grad_main, "loss": loss_main, "diag": diag_main,
             "hvp": hvp_main}
    srcs = {kind: _GHEAD.format(extra_ext=ext, out_bindings=ob, **head_fmt)
            + mains[kind] for kind, (ext, ob) in outs.items()}
    return srcs, np.asarray(values, np.float32)


class GroupedRowTerm(EqRowTerm):
    """Shared-gather JIT kernel for m co-located operators: per point, ONE
    union-site gather feeds every residual (the hand-written-ns5 shape,
    emitted from the operator table). bind_points(x, W(n,m), S(n,m))."""

    def __init__(self, ctx, bases, inv_widths, ops, ids, compiler=None,
                 cache_dir=None):
        compiler = compiler or find_compiler()
        if compiler is None:
            raise RuntimeError("no glslc/glslangValidator available")
        cache_dir = cache_dir or CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self.ids = list(ids)
        self.m = len(self.ids)
        srcs, values = emit_group_shaders(ops, self.ids)
        key = source_key(srcs)
        spv = {kind: os.path.join(cache_dir, f"{key}_{kind}.spv")
               for kind in KINDS}
        for kind, p in spv.items():
            if not valid_spv(p):
                _compile(srcs[kind], p, compiler)
        sub = OperatorTable()
        for k in self.ids:
            sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
        super().__init__(ctx, bases, inv_widths, sub)
        self.grad_program = ctx.program(spv["grad"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.loss_program = ctx.program(spv["loss"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.diag_program = ctx.program(spv["diag"], bindings=[STORAGE] * 11,
                                        spec_constant_ids=[0, 1])
        self.hvp_program = ctx.program(spv["hvp"], bindings=[STORAGE] * 12,
                                       spec_constant_ids=[0, 1])
        self.vals_buf = ctx.buffer(max(values.nbytes, 4))
        self.vals_buf.upload(values)
        self.structure = group_structure_hash(ops, self.ids)
        self.cache_key = key

    def bind_batch(self, *a, **kw):
        raise NotImplementedError(
            "GroupedRowTerm binds with bind_points(x_enc, W(n,m), S(n,m)); "
            "the inherited bind_batch would install a _batch without wb/sb")

    def bind_points(self, x_enc, W, S, scale=1.0):
        ctx = self.ctx
        x_enc = np.ascontiguousarray(x_enc, np.float32)
        n = x_enc.shape[0]
        W = np.ascontiguousarray(W, np.float32).reshape(n, self.m)
        S = np.ascontiguousarray(S, np.float32).reshape(n, self.m)
        meta_b = pack_eqrow_meta(self.bases, n, scale, self.m, 0)
        xb = ctx.buffer(x_enc.nbytes); xb.upload(x_enc.reshape(-1))
        mb = ctx.buffer(len(meta_b), device_local=False); mb.upload(meta_b)
        wb = ctx.buffer(W.nbytes); wb.upload(W.reshape(-1))
        sb = ctx.buffer(S.nbytes); sb.upload(S.reshape(-1))
        self._batch = dict(n=n, xb=xb, mb=mb, wb=wb, sb=sb, scale=scale)
        return self

    def _shared(self, coef_buf):
        b = self._batch
        return [b["mb"], b["xb"], coef_buf, self.coefs_buf, self.table_buf,
                self.dcoefs_buf, self.ddcoefs_buf, b["wb"], b["sb"],
                self.vals_buf]


def verify_grouped(ctx, term, bases, inv_widths, ops, ids, seed=0, n=48,
                   rtol=2e-4, force=False):
    """Parity: grouped term vs generic EqRowTerm on equivalent rows — loss,
    grad, diag AND hvp. Memoized like verify_generated; leaves `term` bound as
    the caller left it."""
    memo = (getattr(term, "cache_key", None), _basis_shape(bases))
    if not force and memo in _VERIFIED:
        return True
    rng = np.random.default_rng(seed)
    nco = int(np.prod([b.primal_extent for b in bases])) * NCH
    xp = verify_points(bases, rng, n)
    m = len(ids)
    W = rng.uniform(0.5, 1.5, (n, m)).astype(np.float32)
    S = (rng.standard_normal((n, m)) * 0.05).astype(np.float32)

    sub = OperatorTable()
    for k in ids:
        sub.add_op(ops.names[k], ops.lin[k], ops.quad[k])
    ref = EqRowTerm(ctx, bases, inv_widths, sub)
    ref.bind_batch(np.repeat(xp, m, axis=0),
                   np.tile(np.arange(m, dtype=np.int32), n),
                   W.reshape(-1), S.reshape(-1), None)
    saved = term._batch
    term.bind_points(xp, W, S)
    try:
        _compare(ref, term, ctx, nco, rtol, "grouped")
    finally:
        term._batch = saved
    _VERIFIED.add(memo)
    return True
