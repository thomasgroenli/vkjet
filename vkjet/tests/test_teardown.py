"""Vulkan teardown is clean: the context-manager path, idempotent destroy(),
transient buffers freed mid-run through __del__, and the atexit path (a child
process that never calls destroy() must exit normally, not segfault)."""
import os
import struct
import subprocess
import sys
import unittest

import numpy as np

from vkjet.context import Context, STORAGE
from vkjet.optim import VEC_AXPBY_SPV


def work(ctx, n=1024):
    """One real dispatch: z = 2·x + 0·y through the vector axpby kernel."""
    a = ctx.buffer(n * 4); a.upload(np.arange(n, dtype=np.float32))
    z = ctx.buffer(n * 4); z.zero()
    vm = ctx.buffer(12, device_local=False); vm.upload(struct.pack("<i2f", n, 2.0, 0.0))
    prog = ctx.program(VEC_AXPBY_SPV, [STORAGE] * 4)
    ctx.run(prog, [a, a, z, vm], groups=(n + 255) // 256)
    return bool(np.allclose(z.download(np.float32, n), 2.0 * np.arange(n)))


def _atexit_child():
    """Runs in a subprocess: a context that is never destroyed explicitly."""
    ctx = Context()
    print("ok" if work(ctx) else "bad")
    sys.exit(0)                         # atexit teardown runs here


class TestTeardown(unittest.TestCase):
    def test_context_manager_and_idempotent_destroy(self):
        with Context() as ctx:
            self.assertTrue(work(ctx))
        ctx.destroy()                   # second destroy is a no-op
        ctx.destroy()

    def test_buffer_churn(self):
        ctx = Context()
        try:
            for _ in range(20):         # 20 × 4 buffers created and GC'd
                self.assertTrue(work(ctx))
        finally:
            ctx.destroy()

    def test_atexit_path(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in [env.get("PYTHONPATH", "")] + sys.path if p)
        r = subprocess.run([sys.executable, "-c",
                            "from vkjet.tests.test_teardown import _atexit_child; _atexit_child()"],
                           capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(r.returncode, 0, f"child exited {r.returncode}\n{r.stderr[-2000:]}")
        self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
