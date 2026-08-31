"""A minimal, synchronous Vulkan compute harness over volkano.

Three objects:

  * :class:`Context` — instance, physical device, compute queue, command
    pool, and a reusable command buffer + fence for synchronous dispatch.
  * :class:`Buffer` — a storage buffer backed by host-visible coherent
    memory, with numpy upload/download. (Device-local + staging can be
    added later when bandwidth matters; host-visible is correct and simple
    for bring-up and validation.)
  * :class:`ComputeProgram` — a SPIR-V module plus its binding signature,
    push-constant size, and specialization constants; compiles and caches
    one pipeline per distinct spec-constant tuple.

The dispatch model is synchronous: :meth:`Context.run` records a one-off
command buffer (bind pipeline, update+bind the program's descriptor set,
push constants, dispatch) and waits on a fence before returning. This makes
the zero-fold / read-after-write ordering of the SplineFlow loop automatic
(§5 of the implementation guide); async overlap is a later optimization.

COMMAND-BUFFER BATCHING: a :class:`CommandSequence` records MANY dispatches /
fills / copies into ONE command buffer (a global memory barrier between
consecutive commands preserves the sequential read-after-write semantics) with
ONE submit + fence — amortizing the ~0.1-0.4 ms per-dispatch launch overhead
that dominates chains of tiny kernels (the GN-CG inner loop). Each recorded
dispatch gets its OWN descriptor set (allocated from the sequence's pool), so
one program may appear many times over different buffers. A recorded sequence
is REUSABLE: submit() it every step as long as the buffers are the same
objects — only their contents may change between submits. Build one either
explicitly (seq.run/fill_zero/copy_buffer) or by capturing existing code:

    seq = ctx.sequence()
    with ctx.capture(seq):
        term.hvp(coef, p, out)      # ctx.run/zero/copy calls append, not execute
    seq.record()
    seq.submit()                    # every step; one fence for the whole chain

During capture, host-visible meta buffers are written immediately (no command)
— so a meta buffer reused with DIFFERENT values across captured dispatches
would silently alias to its final value; give each captured constant its own
buffer. Uploads/downloads of device-local buffers are refused inside capture.
"""
from __future__ import annotations

import atexit
import contextlib
import ctypes
import weakref
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._vk import vk, arrptr, check, enum, Pointer, char, float32, uint32

STORAGE = enum("VK_DESCRIPTOR_TYPE_STORAGE_BUFFER")
UNIFORM = enum("VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER")
COMPUTE_STAGE = enum("VK_SHADER_STAGE_COMPUTE_BIT")
HOST_COHERENT = (
    enum("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT")
    | enum("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT")
)
DEVICE_LOCAL = enum("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT")


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
class Context:
    def __init__(self, prefer_discrete: bool = True, device_index: Optional[int] = None):
        # teardown bookkeeping (set before any Vulkan object is created)
        self._device_valid = False   # True once the device exists, until destroyed
        self._destroyed = False
        self._tracked = weakref.WeakSet()   # live Buffers + ComputePrograms
        self._staging_buf = None            # reused host-visible staging buffer
        self._capture: Optional["CommandSequence"] = None
        self._create_instance()
        self._pick_physical_device(prefer_discrete, device_index)
        self._create_device_and_queue()
        self._device_valid = True
        self._create_command_pool()
        # one reusable command buffer + fence for synchronous dispatch
        self.cmd = self._alloc_command_buffer()
        self.fence = self._create_fence()
        # destroy in the correct order before the interpreter's ctypes
        # finalizers run (which otherwise race the driver → segfault at exit).
        atexit.register(self.destroy)

    # -- teardown ---------------------------------------------------------- #
    def _track(self, obj):
        self._tracked.add(obj)

    def destroy(self):
        """Destroy every Vulkan object in dependency order. Idempotent."""
        if self._destroyed:
            return
        self._destroyed = True
        if self._device_valid:
            try:
                vk.vkDeviceWaitIdle(self.device)
            except Exception:
                pass
            for obj in list(self._tracked):   # buffers + programs (children)
                try:
                    obj.destroy()
                except Exception:
                    pass
            for fn, h in ((vk.vkDestroyFence, self.fence),
                          (vk.vkDestroyCommandPool, self.command_pool)):
                try:
                    fn(self.device, h, None)
                except Exception:
                    pass
            self._device_valid = False
            try:
                vk.vkDestroyDevice(self.device, None)
            except Exception:
                pass
        try:
            vk.vkDestroyInstance(self.instance, None)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    # -- setup ------------------------------------------------------------- #
    def _create_instance(self):
        api_1_2 = (1 << 22) | (2 << 12)  # VK_API_VERSION_1_2
        app = vk.VkApplicationInfo()
        app.sType = enum("VK_STRUCTURE_TYPE_APPLICATION_INFO")
        app.pNext = None; app.pApplicationName = None; app.applicationVersion = 0
        app.pEngineName = None; app.engineVersion = 0; app.apiVersion = api_1_2
        self._app_keepalive = app
        ci = vk.VkInstanceCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO")
        ci.pNext = None; ci.flags = 0; ci.pApplicationInfo = app.ptr
        ci.enabledLayerCount = 0; ci.ppEnabledLayerNames = None
        ci.enabledExtensionCount = 0; ci.ppEnabledExtensionNames = None
        self.instance = vk.VkInstance()
        check(vk.vkCreateInstance(ci.ref, None, self.instance.ref), "vkCreateInstance")

    def _pick_physical_device(self, prefer_discrete, device_index):
        n = vk.uint32_t(0)
        vk.vkEnumeratePhysicalDevices(self.instance, n.ref, None)
        devs = (vk.VkPhysicalDevice * int(n.value))()
        vk.vkEnumeratePhysicalDevices(self.instance, n.ref, arrptr(devs, vk.VkPhysicalDevice))
        chosen = 0
        if device_index is not None:
            chosen = device_index
        elif prefer_discrete:
            for i in range(int(n.value)):
                p = vk.VkPhysicalDeviceProperties()
                vk.vkGetPhysicalDeviceProperties(devs[i], p.ref)
                if int(p.deviceType) == 2:  # DISCRETE_GPU
                    chosen = i
                    break
        self.physical_device = devs[chosen]
        props = vk.VkPhysicalDeviceProperties()
        vk.vkGetPhysicalDeviceProperties(self.physical_device, props.ref)
        self.device_name = bytes(props.deviceName).split(b"\x00")[0].decode()
        self.limits = props.limits
        self.subgroup_size = 32  # NVIDIA warp; queried properly later if needed
        self.memprops = vk.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, self.memprops.ref)

    def _create_device_and_queue(self):
        qn = vk.uint32_t(0)
        vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device, qn.ref, None)
        qf = (vk.VkQueueFamilyProperties * int(qn.value))()
        vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self.physical_device, qn.ref, arrptr(qf, vk.VkQueueFamilyProperties))
        compute_bit = enum("VK_QUEUE_COMPUTE_BIT")
        self.queue_family = next(
            i for i in range(int(qn.value)) if int(qf[i].queueFlags) & compute_bit)

        prio = (ctypes.c_float * 1)(1.0)
        qci = vk.VkDeviceQueueCreateInfo()
        qci.sType = enum("VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO")
        qci.pNext = None; qci.flags = 0
        qci.queueFamilyIndex = self.queue_family; qci.queueCount = 1
        qci.pQueuePriorities = ctypes.cast(prio, Pointer[float32])
        self._prio_keepalive = prio

        # scalarBlockLayout: the sort shaders declare std430 UBOs
        # (GL_EXT_scalar_block_layout) so int[16] arrays pack at 4-byte stride.
        feats12 = vk.VkPhysicalDeviceVulkan12Features()
        feats12.sType = enum("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES")
        feats12.pNext = None
        feats12.scalarBlockLayout = 1
        self._feats12_keepalive = feats12

        # VK_EXT_shader_atomic_float: native atomicAdd(float) in the tuned
        # scatter kernels (assumed present). Chained after feats12.
        atomicf = vk.VkPhysicalDeviceShaderAtomicFloatFeaturesEXT()
        atomicf.sType = enum("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT")
        atomicf.pNext = None
        atomicf.shaderBufferFloat32AtomicAdd = 1
        self._atomicf_keepalive = atomicf
        feats12.pNext = atomicf.ptr.cast(Pointer[None])

        ext_names = (ctypes.c_char_p * 1)(b"VK_EXT_shader_atomic_float")
        self._ext_keepalive = ext_names

        dci = vk.VkDeviceCreateInfo()
        dci.sType = enum("VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO")
        dci.pNext = feats12.ptr.cast(Pointer[None]); dci.flags = 0
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = qci.ptr
        dci.enabledLayerCount = 0; dci.ppEnabledLayerNames = None
        dci.enabledExtensionCount = 1
        dci.ppEnabledExtensionNames = ctypes.cast(ext_names, Pointer[Pointer[char]])
        dci.pEnabledFeatures = None
        self.device = vk.VkDevice()
        check(vk.vkCreateDevice(self.physical_device, dci.ref, None, self.device.ref),
              "vkCreateDevice")
        self.queue = vk.VkQueue()
        vk.vkGetDeviceQueue(self.device, self.queue_family, 0, self.queue.ref)

    def _create_command_pool(self):
        ci = vk.VkCommandPoolCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO")
        ci.pNext = None
        ci.flags = enum("VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT")
        ci.queueFamilyIndex = self.queue_family
        self.command_pool = vk.VkCommandPool()
        check(vk.vkCreateCommandPool(self.device, ci.ref, None, self.command_pool.ref),
              "vkCreateCommandPool")

    def _alloc_command_buffer(self):
        ai = vk.VkCommandBufferAllocateInfo()
        ai.sType = enum("VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO")
        ai.pNext = None; ai.commandPool = self.command_pool
        ai.level = enum("VK_COMMAND_BUFFER_LEVEL_PRIMARY"); ai.commandBufferCount = 1
        cmd = vk.VkCommandBuffer()
        check(vk.vkAllocateCommandBuffers(self.device, ai.ref, cmd.ref),
              "vkAllocateCommandBuffers")
        return cmd

    def _create_fence(self):
        ci = vk.VkFenceCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_FENCE_CREATE_INFO")
        ci.pNext = None; ci.flags = 0
        f = vk.VkFence()
        check(vk.vkCreateFence(self.device, ci.ref, None, f.ref), "vkCreateFence")
        return f

    # -- memory ------------------------------------------------------------ #
    def find_memory_type(self, type_bits, want):
        for i in range(int(self.memprops.memoryTypeCount)):
            flags = int(self.memprops.memoryTypes[i].propertyFlags)
            if (type_bits >> i) & 1 and (flags & want) == want:
                return i
        raise RuntimeError(f"no memory type for bits={type_bits:#x} want={want:#x}")

    # -- factories --------------------------------------------------------- #
    def buffer(self, nbytes, device_local=True, usage_extra=0):
        return Buffer(self, nbytes, device_local, usage_extra)

    def program(self, spv_path_or_bytes, bindings, push_constant_bytes=0,
                spec_constant_ids=()):
        return ComputeProgram(self, spv_path_or_bytes, bindings,
                              push_constant_bytes, spec_constant_ids)

    # -- dispatch (synchronous) ------------------------------------------- #
    def run(self, program, buffers, groups, spec=None, push=None):
        """Bind ``program`` over ``buffers`` and dispatch ``groups`` workgroups.

        ``groups`` is an int (x) or a 3-tuple. ``spec`` selects the pipeline
        variant (dict {constant_id: int}). ``push`` is raw bytes for the push
        constant block. Blocks until the dispatch completes (fence wait).
        """
        if self._capture is not None:
            assert push is None, "push constants are not supported inside capture"
            self._capture.run(program, buffers, groups, spec)
            return
        if isinstance(groups, int):
            gx, gy, gz = groups, 1, 1
        else:
            gx, gy, gz = (list(groups) + [1, 1])[:3]
        pipeline = program.pipeline(spec or {})
        program.update_descriptors(buffers)

        bind_pt = enum("VK_PIPELINE_BIND_POINT_COMPUTE")
        push_keep = []
        if push is not None:
            buf = (ctypes.c_char * len(push)).from_buffer_copy(push)
            push_keep.append(buf)

        def record(cmd):
            vk.vkCmdBindPipeline(cmd, bind_pt, pipeline)
            vk.vkCmdBindDescriptorSets(cmd, bind_pt, program.layout, 0, 1,
                                       program.descriptor_set.ref, 0, None)
            if push is not None:
                vk.vkCmdPushConstants(cmd, program.layout, COMPUTE_STAGE, 0, len(push),
                                      ctypes.cast(push_keep[0], Pointer[None]))
            vk.vkCmdDispatch(cmd, gx, gy, gz)
        self._oneshot(record)

    # -- one-shot command submit (used by run + buffer transfers) --------- #
    def _oneshot(self, record_fn):
        cmd = self.cmd
        bi = vk.VkCommandBufferBeginInfo()
        bi.sType = enum("VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO")
        bi.pNext = None
        bi.flags = enum("VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT")
        bi.pInheritanceInfo = None
        check(vk.vkBeginCommandBuffer(cmd, bi.ref), "vkBeginCommandBuffer")
        record_fn(cmd)
        check(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")
        check(vk.vkResetFences(self.device, 1, self.fence.ref), "vkResetFences")
        si = vk.VkSubmitInfo()
        si.sType = enum("VK_STRUCTURE_TYPE_SUBMIT_INFO"); si.pNext = None
        si.waitSemaphoreCount = 0; si.pWaitSemaphores = None; si.pWaitDstStageMask = None
        si.commandBufferCount = 1; si.pCommandBuffers = cmd.ptr
        si.signalSemaphoreCount = 0; si.pSignalSemaphores = None
        check(vk.vkQueueSubmit(self.queue, 1, si.ref, self.fence), "vkQueueSubmit")
        check(vk.vkWaitForFences(self.device, 1, self.fence.ref, 1, 30_000_000_000),
              "vkWaitForFences")

    # -- staging transfers for device-local buffers ----------------------- #
    def _staging(self, nbytes):
        if self._staging_buf is None or self._staging_buf.nbytes < nbytes:
            self._staging_buf = Buffer(self, max(nbytes, 1 << 20), device_local=False)
        return self._staging_buf

    def copy_buffer(self, src, dst, nbytes):
        if self._capture is not None:
            assert src is not self._staging_buf and dst is not self._staging_buf, \
                "staging transfers (upload/download) are not allowed inside capture"
            self._capture.copy_buffer(src, dst, nbytes)
            return
        region = vk.VkBufferCopy(); region.srcOffset = 0; region.dstOffset = 0
        region.size = nbytes
        self._oneshot(lambda cmd: vk.vkCmdCopyBuffer(cmd, src.handle, dst.handle, 1, region.ref))

    def fill_zero(self, buf):
        if self._capture is not None:
            self._capture.fill_zero(buf)
            return
        self._oneshot(lambda cmd: vk.vkCmdFillBuffer(cmd, buf.handle, 0, buf.nbytes, 0))

    # -- command-buffer batching ------------------------------------------- #
    def sequence(self):
        """A reusable multi-dispatch command buffer (see module docstring)."""
        return CommandSequence(self)

    @contextlib.contextmanager
    def capture(self, seq: "CommandSequence"):
        """Redirect ctx.run / Buffer.zero / copy_buffer into ``seq`` (append,
        don't execute) so existing term code can be batched unmodified."""
        assert self._capture is None, "capture is not reentrant"
        self._capture = seq
        try:
            yield seq
        finally:
            self._capture = None


# --------------------------------------------------------------------------- #
# Buffer
# --------------------------------------------------------------------------- #
class Buffer:
    """A storage buffer. ``device_local=True`` (default) puts it in VRAM —
    essential for compute throughput: atomics and gather/scatter on a
    host-visible (system-RAM/PCIe) buffer run ~100× slower. Upload/download
    go through a reused host-visible staging buffer + vkCmdCopyBuffer.
    ``device_local=False`` is host-mappable (used for staging itself)."""

    def __init__(self, ctx: Context, nbytes: int, device_local: bool = True,
                 usage_extra: int = 0):
        self.ctx = ctx
        self._destroyed = False
        self.nbytes = int(nbytes)
        self.device_local = device_local
        usage = (enum("VK_BUFFER_USAGE_STORAGE_BUFFER_BIT")
                 | enum("VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT")
                 | enum("VK_BUFFER_USAGE_TRANSFER_SRC_BIT")
                 | enum("VK_BUFFER_USAGE_TRANSFER_DST_BIT") | usage_extra)
        bci = vk.VkBufferCreateInfo()
        bci.sType = enum("VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO")
        bci.pNext = None; bci.flags = 0; bci.size = self.nbytes
        bci.usage = usage; bci.sharingMode = enum("VK_SHARING_MODE_EXCLUSIVE")
        bci.queueFamilyIndexCount = 0; bci.pQueueFamilyIndices = None
        self.handle = vk.VkBuffer()
        check(vk.vkCreateBuffer(ctx.device, bci.ref, None, self.handle.ref), "vkCreateBuffer")
        req = vk.VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(ctx.device, self.handle, req.ref)
        self.alloc_size = int(req.size)
        want = DEVICE_LOCAL if device_local else HOST_COHERENT
        ai = vk.VkMemoryAllocateInfo()
        ai.sType = enum("VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO")
        ai.pNext = None; ai.allocationSize = self.alloc_size
        ai.memoryTypeIndex = ctx.find_memory_type(int(req.memoryTypeBits), want)
        self.memory = vk.VkDeviceMemory()
        check(vk.vkAllocateMemory(ctx.device, ai.ref, None, self.memory.ref), "vkAllocateMemory")
        check(vk.vkBindBufferMemory(ctx.device, self.handle, self.memory, 0), "vkBindBufferMemory")
        ctx._track(self)

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        if self.ctx._device_valid:
            vk.vkDestroyBuffer(self.ctx.device, self.handle, None)
            vk.vkFreeMemory(self.ctx.device, self.memory, None)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    # -- host-visible path (valid only when not device_local) ------------- #
    def _host_write(self, raw):
        p = Pointer[None]()
        check(vk.vkMapMemory(self.ctx.device, self.memory, 0, len(raw), 0, p.ptr), "vkMapMemory")
        src = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
        ctypes.memmove(p, src, len(raw))
        vk.vkUnmapMemory(self.ctx.device, self.memory)

    def _host_read(self, n):
        p = Pointer[None]()
        check(vk.vkMapMemory(self.ctx.device, self.memory, 0, n, 0, p.ptr), "vkMapMemory")
        out = (ctypes.c_char * n)()
        ctypes.memmove(out, p, n)
        vk.vkUnmapMemory(self.ctx.device, self.memory)
        return bytes(out)

    # -- public API (device-local aware) ---------------------------------- #
    def upload(self, data):
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        assert len(raw) <= self.nbytes, f"upload {len(raw)} > buffer {self.nbytes}"
        if not self.device_local:
            self._host_write(raw)
        else:
            st = self.ctx._staging(len(raw))
            st._host_write(raw)
            self.ctx.copy_buffer(st, self, len(raw))

    def download(self, dtype=np.float32, count=None):
        n = self.nbytes if count is None else count * np.dtype(dtype).itemsize
        if not self.device_local:
            raw = self._host_read(n)
        else:
            st = self.ctx._staging(n)
            self.ctx.copy_buffer(self, st, n)
            raw = st._host_read(n)
        return np.frombuffer(raw, dtype=dtype)

    def zero(self):
        if not self.device_local:
            p = Pointer[None]()
            check(vk.vkMapMemory(self.ctx.device, self.memory, 0, self.nbytes, 0, p.ptr), "vkMapMemory")
            ctypes.memset(p, 0, self.nbytes)
            vk.vkUnmapMemory(self.ctx.device, self.memory)
        else:
            self.ctx.fill_zero(self)


# --------------------------------------------------------------------------- #
# ComputeProgram
# --------------------------------------------------------------------------- #
class ComputeProgram:
    def __init__(self, ctx: Context, spv, bindings: Sequence[int],
                 push_constant_bytes: int = 0, spec_constant_ids: Sequence[int] = ()):
        self.ctx = ctx
        self._destroyed = False
        self.bindings = list(bindings)          # descriptor type per binding
        self.push_constant_bytes = int(push_constant_bytes)
        self.spec_constant_ids = list(spec_constant_ids)
        self._pipelines: Dict[tuple, object] = {}

        spv_bytes = spv if isinstance(spv, (bytes, bytearray)) else open(spv, "rb").read()
        self.module = self._make_shader_module(spv_bytes)
        self.dsl = self._make_descriptor_set_layout()
        self.layout = self._make_pipeline_layout()
        self._make_descriptor_set()
        ctx._track(self)

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        if not self.ctx._device_valid:
            return
        dev = self.ctx.device
        for pipe in self._pipelines.values():
            vk.vkDestroyPipeline(dev, pipe, None)
        vk.vkDestroyDescriptorPool(dev, self.pool, None)     # frees its sets
        vk.vkDestroyPipelineLayout(dev, self.layout, None)
        vk.vkDestroyDescriptorSetLayout(dev, self.dsl, None)
        vk.vkDestroyShaderModule(dev, self.module, None)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def _make_shader_module(self, spv_bytes):
        ci = vk.VkShaderModuleCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO")
        ci.pNext = None; ci.flags = 0; ci.codeSize = len(spv_bytes)
        code = (ctypes.c_uint32 * (len(spv_bytes) // 4)).from_buffer_copy(spv_bytes)
        self._code_keepalive = code
        ci.pCode = arrptr(code, uint32)
        mod = vk.VkShaderModule()
        check(vk.vkCreateShaderModule(self.ctx.device, ci.ref, None, mod.ref),
              "vkCreateShaderModule")
        return mod

    def _make_descriptor_set_layout(self):
        n = len(self.bindings)
        binds = (vk.VkDescriptorSetLayoutBinding * n)()
        for i, dtype in enumerate(self.bindings):
            binds[i].binding = i
            binds[i].descriptorType = dtype
            binds[i].descriptorCount = 1
            binds[i].stageFlags = COMPUTE_STAGE
            binds[i].pImmutableSamplers = None
        self._binds_keepalive = binds
        ci = vk.VkDescriptorSetLayoutCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO")
        ci.pNext = None; ci.flags = 0; ci.bindingCount = n
        ci.pBindings = arrptr(binds, vk.VkDescriptorSetLayoutBinding)
        dsl = vk.VkDescriptorSetLayout()
        check(vk.vkCreateDescriptorSetLayout(self.ctx.device, ci.ref, None, dsl.ref), "DSL")
        return dsl

    def _make_pipeline_layout(self):
        ci = vk.VkPipelineLayoutCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO")
        ci.pNext = None; ci.flags = 0; ci.setLayoutCount = 1; ci.pSetLayouts = self.dsl.ptr
        if self.push_constant_bytes:
            pcr = vk.VkPushConstantRange()
            pcr.stageFlags = COMPUTE_STAGE; pcr.offset = 0; pcr.size = self.push_constant_bytes
            self._pcr_keepalive = pcr
            ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = pcr.ptr
        else:
            ci.pushConstantRangeCount = 0; ci.pPushConstantRanges = None
        lo = vk.VkPipelineLayout()
        check(vk.vkCreatePipelineLayout(self.ctx.device, ci.ref, None, lo.ref), "PLL")
        return lo

    def _make_descriptor_set(self):
        # one pool, one persistent set; re-updated each dispatch (safe because
        # dispatches are synchronous — we wait before reusing).
        counts: Dict[int, int] = {}
        for d in self.bindings:
            counts[d] = counts.get(d, 0) + 1
        sizes = (vk.VkDescriptorPoolSize * len(counts))()
        for i, (dtype, cnt) in enumerate(counts.items()):
            sizes[i].type = dtype; sizes[i].descriptorCount = cnt
        self._sizes_keepalive = sizes
        pci = vk.VkDescriptorPoolCreateInfo()
        pci.sType = enum("VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO")
        pci.pNext = None; pci.flags = 0; pci.maxSets = 1
        pci.poolSizeCount = len(counts); pci.pPoolSizes = arrptr(sizes, vk.VkDescriptorPoolSize)
        self.pool = vk.VkDescriptorPool()
        check(vk.vkCreateDescriptorPool(self.ctx.device, pci.ref, None, self.pool.ref), "pool")
        ai = vk.VkDescriptorSetAllocateInfo()
        ai.sType = enum("VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO")
        ai.pNext = None; ai.descriptorPool = self.pool; ai.descriptorSetCount = 1
        ai.pSetLayouts = self.dsl.ptr
        self.descriptor_set = vk.VkDescriptorSet()
        check(vk.vkAllocateDescriptorSets(self.ctx.device, ai.ref, self.descriptor_set.ref),
              "vkAllocateDescriptorSets")

    def update_descriptors(self, buffers):
        assert len(buffers) == len(self.bindings), \
            f"expected {len(self.bindings)} buffers, got {len(buffers)}"
        n = len(buffers)
        infos = (vk.VkDescriptorBufferInfo * n)()
        writes = (vk.VkWriteDescriptorSet * n)()
        for i, b in enumerate(buffers):
            infos[i].buffer = b.handle; infos[i].offset = 0
            infos[i].range = enum("VK_WHOLE_SIZE")
            writes[i].sType = enum("VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET")
            writes[i].pNext = None; writes[i].dstSet = self.descriptor_set
            writes[i].dstBinding = i; writes[i].dstArrayElement = 0
            writes[i].descriptorCount = 1; writes[i].descriptorType = self.bindings[i]
            writes[i].pBufferInfo = ctypes.cast(
                ctypes.byref(infos, ctypes.sizeof(vk.VkDescriptorBufferInfo) * i),
                Pointer[vk.VkDescriptorBufferInfo])
            writes[i].pImageInfo = None; writes[i].pTexelBufferView = None
        self._infos_keepalive = infos
        vk.vkUpdateDescriptorSets(self.ctx.device, n,
                                  arrptr(writes, vk.VkWriteDescriptorSet), 0, None)

    def pipeline(self, spec: Dict[int, int]):
        key = tuple(int(spec.get(cid, 0)) for cid in self.spec_constant_ids)
        if key in self._pipelines:
            return self._pipelines[key]
        pipe = self._compile_pipeline(spec)
        self._pipelines[key] = pipe
        return pipe

    def _compile_pipeline(self, spec):
        stage = vk.VkPipelineShaderStageCreateInfo()
        stage.sType = enum("VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO")
        stage.pNext = None; stage.flags = 0; stage.stage = COMPUTE_STAGE
        stage.module = self.module
        name = ctypes.create_string_buffer(b"main")
        stage.pName = ctypes.cast(name, Pointer[char])

        spec_info = None
        if self.spec_constant_ids:
            ids = self.spec_constant_ids
            entries = (vk.VkSpecializationMapEntry * len(ids))()
            data = (ctypes.c_int32 * len(ids))()
            for i, cid in enumerate(ids):
                entries[i].constantID = cid
                entries[i].offset = i * 4
                entries[i].size = 4
                data[i] = int(spec.get(cid, 0))
            spec_info = vk.VkSpecializationInfo()
            spec_info.mapEntryCount = len(ids)
            spec_info.pMapEntries = arrptr(entries, vk.VkSpecializationMapEntry)
            spec_info.dataSize = len(ids) * 4
            spec_info.pData = ctypes.cast(data, Pointer[None])
            stage.pSpecializationInfo = spec_info.ptr
            self._spec_keepalive = (entries, data, spec_info, name)
        else:
            stage.pSpecializationInfo = None
            self._spec_keepalive = (name,)

        ci = vk.VkComputePipelineCreateInfo()
        ci.sType = enum("VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO")
        ci.pNext = None; ci.flags = 0; ci.stage = stage; ci.layout = self.layout
        ci.basePipelineHandle = vk.VkPipeline(); ci.basePipelineIndex = -1
        pipe = vk.VkPipeline()
        check(vk.vkCreateComputePipelines(self.ctx.device, vk.VkPipelineCache(), 1,
                                          ci.ref, None, pipe.ref), "vkCreateComputePipelines")
        return pipe


# --------------------------------------------------------------------------- #
# CommandSequence — many dispatches, one submit
# --------------------------------------------------------------------------- #
class CommandSequence:
    """Records dispatches/fills/copies into one reusable command buffer.

    Build (directly or via ctx.capture), then record() once, then submit() as
    many times as desired — the recorded work re-executes against the SAME
    buffer objects (contents may change between submits; bindings may not).
    reset() clears everything for a fresh build. A conservative global memory
    barrier between consecutive commands reproduces the sequential
    read-after-write semantics of the one-dispatch-per-submit path."""

    _SRC_ACCESS = None  # filled lazily (enum lookups)

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._destroyed = False
        self.steps: List[tuple] = []
        self.cmd = ctx._alloc_command_buffer()
        self.pool = None
        self._pool_sets = 0
        self._recorded = False
        self._keep = []
        ctx._track(self)

    # -- building ----------------------------------------------------------- #
    def run(self, program, buffers, groups, spec=None):
        assert not self._recorded, "reset() before adding to a recorded sequence"
        if isinstance(groups, int):
            groups = (groups, 1, 1)
        else:
            groups = tuple((list(groups) + [1, 1])[:3])
        self.steps.append(("dispatch", program, list(buffers), groups, dict(spec or {})))

    def fill_zero(self, buf):
        assert not self._recorded
        self.steps.append(("fill", buf))

    def copy_buffer(self, src, dst, nbytes):
        assert not self._recorded
        self.steps.append(("copy", src, dst, int(nbytes)))

    # -- recording ---------------------------------------------------------- #
    def _ensure_pool(self, n_sets, type_counts):
        if self.pool is not None and self._pool_sets >= n_sets:
            check(vk.vkResetDescriptorPool(self.ctx.device, self.pool, 0),
                  "vkResetDescriptorPool")
            return
        if self.pool is not None:
            vk.vkDestroyDescriptorPool(self.ctx.device, self.pool, None)
        sizes = (vk.VkDescriptorPoolSize * max(len(type_counts), 1))()
        for i, (dtype, cnt) in enumerate(type_counts.items()):
            sizes[i].type = dtype; sizes[i].descriptorCount = cnt
        self._keep.append(sizes)
        pci = vk.VkDescriptorPoolCreateInfo()
        pci.sType = enum("VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO")
        pci.pNext = None; pci.flags = 0; pci.maxSets = max(n_sets, 1)
        pci.poolSizeCount = max(len(type_counts), 1)
        pci.pPoolSizes = arrptr(sizes, vk.VkDescriptorPoolSize)
        self.pool = vk.VkDescriptorPool()
        check(vk.vkCreateDescriptorPool(self.ctx.device, pci.ref, None, self.pool.ref),
              "sequence descriptor pool")
        self._pool_sets = max(n_sets, 1)

    def _barrier(self, cmd):
        mb = vk.VkMemoryBarrier()
        mb.sType = enum("VK_STRUCTURE_TYPE_MEMORY_BARRIER")
        mb.pNext = None
        mb.srcAccessMask = (enum("VK_ACCESS_SHADER_WRITE_BIT")
                            | enum("VK_ACCESS_TRANSFER_WRITE_BIT"))
        mb.dstAccessMask = (enum("VK_ACCESS_SHADER_READ_BIT")
                            | enum("VK_ACCESS_SHADER_WRITE_BIT")
                            | enum("VK_ACCESS_TRANSFER_READ_BIT")
                            | enum("VK_ACCESS_TRANSFER_WRITE_BIT"))
        self._keep.append(mb)
        stages = (enum("VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT")
                  | enum("VK_PIPELINE_STAGE_TRANSFER_BIT"))
        vk.vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, mb.ref, 0, None, 0, None)

    def record(self):
        """Allocate one descriptor set per dispatch, then record the whole chain."""
        assert not self._recorded
        self._keep = []
        dispatches = [s for s in self.steps if s[0] == "dispatch"]
        type_counts: Dict[int, int] = {}
        for _, prog, bufs, _, _ in dispatches:
            for d in prog.bindings:
                type_counts[d] = type_counts.get(d, 0) + 1
        self._ensure_pool(len(dispatches), type_counts)
        dsets = []
        for _, prog, bufs, _, _ in dispatches:
            ai = vk.VkDescriptorSetAllocateInfo()
            ai.sType = enum("VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO")
            ai.pNext = None; ai.descriptorPool = self.pool
            ai.descriptorSetCount = 1; ai.pSetLayouts = prog.dsl.ptr
            ds = vk.VkDescriptorSet()
            check(vk.vkAllocateDescriptorSets(self.ctx.device, ai.ref, ds.ref),
                  "sequence descriptor set")
            _write_descriptor_set(self.ctx, ds, prog.bindings, bufs)
            dsets.append(ds)
            self._keep.append(ds)

        bind_pt = enum("VK_PIPELINE_BIND_POINT_COMPUTE")
        bi = vk.VkCommandBufferBeginInfo()
        bi.sType = enum("VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO")
        bi.pNext = None; bi.flags = 0; bi.pInheritanceInfo = None
        check(vk.vkBeginCommandBuffer(self.cmd, bi.ref), "vkBeginCommandBuffer")
        di = 0
        for k, step in enumerate(self.steps):
            if k:
                self._barrier(self.cmd)
            if step[0] == "dispatch":
                _, prog, bufs, groups, spec = step
                vk.vkCmdBindPipeline(self.cmd, bind_pt, prog.pipeline(spec))
                vk.vkCmdBindDescriptorSets(self.cmd, bind_pt, prog.layout, 0, 1,
                                           dsets[di].ref, 0, None)
                vk.vkCmdDispatch(self.cmd, *groups)
                di += 1
            elif step[0] == "fill":
                buf = step[1]
                vk.vkCmdFillBuffer(self.cmd, buf.handle, 0, buf.nbytes, 0)
            else:
                _, src, dst, nbytes = step
                region = vk.VkBufferCopy()
                region.srcOffset = 0; region.dstOffset = 0; region.size = nbytes
                self._keep.append(region)
                vk.vkCmdCopyBuffer(self.cmd, src.handle, dst.handle, 1, region.ref)
        check(vk.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")
        self._recorded = True
        return self

    # -- execution ---------------------------------------------------------- #
    def submit(self):
        """Submit the recorded chain once and wait (one fence for everything)."""
        assert self._recorded, "record() before submit()"
        ctx = self.ctx
        check(vk.vkResetFences(ctx.device, 1, ctx.fence.ref), "vkResetFences")
        si = vk.VkSubmitInfo()
        si.sType = enum("VK_STRUCTURE_TYPE_SUBMIT_INFO"); si.pNext = None
        si.waitSemaphoreCount = 0; si.pWaitSemaphores = None; si.pWaitDstStageMask = None
        si.commandBufferCount = 1; si.pCommandBuffers = self.cmd.ptr
        si.signalSemaphoreCount = 0; si.pSignalSemaphores = None
        check(vk.vkQueueSubmit(ctx.queue, 1, si.ref, ctx.fence), "vkQueueSubmit")
        check(vk.vkWaitForFences(ctx.device, 1, ctx.fence.ref, 1, 30_000_000_000),
              "vkWaitForFences")

    def reset(self):
        """Discard the recorded chain (descriptor pool is recycled on re-record)."""
        self.steps = []
        self._recorded = False
        self._keep = []

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        if self.ctx._device_valid:
            if self.pool is not None:
                vk.vkDestroyDescriptorPool(self.ctx.device, self.pool, None)
            vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1,
                                    self.cmd.ptr)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


def _write_descriptor_set(ctx, dset, bindings, buffers):
    """Point ``dset``'s bindings at ``buffers`` (immediate copy; no keepalive)."""
    assert len(buffers) == len(bindings), \
        f"expected {len(bindings)} buffers, got {len(buffers)}"
    n = len(buffers)
    infos = (vk.VkDescriptorBufferInfo * n)()
    writes = (vk.VkWriteDescriptorSet * n)()
    for i, b in enumerate(buffers):
        infos[i].buffer = b.handle; infos[i].offset = 0
        infos[i].range = enum("VK_WHOLE_SIZE")
        writes[i].sType = enum("VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET")
        writes[i].pNext = None; writes[i].dstSet = dset
        writes[i].dstBinding = i; writes[i].dstArrayElement = 0
        writes[i].descriptorCount = 1; writes[i].descriptorType = bindings[i]
        writes[i].pBufferInfo = ctypes.cast(
            ctypes.byref(infos, ctypes.sizeof(vk.VkDescriptorBufferInfo) * i),
            Pointer[vk.VkDescriptorBufferInfo])
        writes[i].pImageInfo = None; writes[i].pTexelBufferView = None
    vk.vkUpdateDescriptorSets(ctx.device, n,
                              arrptr(writes, vk.VkWriteDescriptorSet), 0, None)
