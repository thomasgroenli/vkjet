"""Registry singleton + low-level ctypes helpers for volkano.

volkano exposes Vulkan as a lazy ctypes registry. This module builds it
once (bound to the system loader) and provides the handful of coercion
helpers the rest of vkjet needs. The idioms encoded here were validated
end-to-end against a real driver (see scratch/smoke_dispatch.py):

  * Scalars / ``Pointer`` live in ``volkano.cbase``, not the registry.
  * Arrays: ``(vk.T * n)()``; pass/fill via ``ctypes.cast(arr, Pointer[T])``;
    indexing an array returns a typed instance (do not re-wrap).
  * Function out-params take ``struct.ref`` (byref); **struct pointer
    fields** must be assigned ``struct.ptr`` (a typed ``Pointer`` instance).
  * ``void**`` out-params (e.g. vkMapMemory) take a ``Pointer[None]()`` via
    its ``.ptr``.
"""
from __future__ import annotations

import ctypes

from volkano.vulkan_stdlib import build_registry
from volkano.cbase import Pointer, float32, uint32, int32, char  # noqa: F401

vk = build_registry(library=True)


def arrptr(arr, elem_type):
    """Pointer to the first element of a ctypes array, typed ``Pointer[T]``."""
    return ctypes.cast(arr, Pointer[elem_type])


def check(result, what):
    """Raise on a non-VK_SUCCESS VkResult."""
    code = int(result)
    if code != 0:
        raise RuntimeError(f"{what} failed: VkResult {code}")


def enum(name):
    """Int value of a Vulkan enum/flag constant by name."""
    return int(getattr(vk, name))
