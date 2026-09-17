"""vkjet's test suite, shipped with the package.

    python -m vkjet.tests            # everything
    python -m vkjet.tests -v         # verbose
    python -m vkjet.tests test_eqrow test_genkernel.TestGenKernel.test_per_op_parity

Every module is a plain `unittest` module and can also be run on its own
(`python -m vkjet.tests.test_eqrow`) or through `python -m unittest`.

One Vulkan context is shared by the whole run (`context()`), created on first
use and destroyed at interpreter exit; a test that needs its own context (the
teardown test) makes one. Tests that need a GLSL compiler for the JIT tier are
skipped, not failed, when none is found (`needs_compiler`).
"""
from __future__ import annotations

import atexit
import os
import tempfile
import unittest

import numpy as np

_CTX = None


def context():
    """The process-wide Context, created lazily. GPU=<index> picks the device."""
    global _CTX
    if _CTX is None:
        from vkjet.context import Context
        gpu = os.environ.get("GPU")
        _CTX = Context(device_index=int(gpu) if gpu else None)
        atexit.register(_CTX.destroy)
    return _CTX


def has_compiler():
    from vkjet.genkernel import find_compiler
    return find_compiler() is not None


def needs_compiler(obj):
    """Skip a test (or a whole TestCase) without a GLSL compiler."""
    return unittest.skipUnless(has_compiler(), "no glslc/glslangValidator on this machine")(obj)


def scratch_dir():
    """A fresh throwaway directory (JIT cache, row files)."""
    return tempfile.mkdtemp(prefix="vkjet-tests-")


def rel(a, b):
    """Relative L2 distance of a from b (scalars or arrays)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
