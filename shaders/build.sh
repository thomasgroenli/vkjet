#!/usr/bin/env bash
# Compile vkjet's static GLSL to SPIR-V (the JIT kernels are generated at run
# time by vkjet/genkernel.py and cached separately, under ~/.cache/vkjet). Needs a compiler that supports
# GL_EXT_shader_atomic_float + subgroup ops: either glslc (shaderc) or a modern
# glslangValidator (≥ ~10; the micromamba py314 env ships 16.x — the old system
# glslangValidator 8.13 is too old). Targets Vulkan 1.1 (SPIR-V 1.3) for subgroup
# intrinsics and spec-constant workgroup size (local_size_x_id).
set -e
cd "$(dirname "$0")"
mkdir -p spv

GLSLC="${GLSLC:-}"
GLSLANG="${GLSLANG:-}"
[ -z "$GLSLC" ] && command -v glslc >/dev/null 2>&1 && GLSLC=glslc
if [ -z "$GLSLC" ] && [ -z "$GLSLANG" ]; then
    for cand in "$HOME/micromamba/envs/py314/bin/glslangValidator" \
                "$(command -v glslangValidator 2>/dev/null)"; do
        [ -x "$cand" ] && GLSLANG="$cand" && break
    done
fi
[ -z "$GLSLC" ] && [ -z "$GLSLANG" ] && { echo "no glslc/glslangValidator found"; exit 1; }

for src in *.comp.glsl; do
    out="spv/${src%.comp.glsl}.spv"
    if [ -n "$GLSLC" ]; then
        "$GLSLC" -fshader-stage=compute --target-env=vulkan1.1 "$src" -o "$out"
    else
        "$GLSLANG" -V --target-env vulkan1.1 -S comp -o "$out" "$src" > /dev/null
    fi
    echo "  $src -> $out ($(stat -c%s "$out") bytes)"
done
echo "done."
