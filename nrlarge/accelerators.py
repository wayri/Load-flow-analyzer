"""Optional FP64 GPU CSR products. Requested backends never fall back to CPU.

OpenCL uses the system ICD through the Khronos C ABI, without pyopencl. OpenGL
uses a hidden 4.3 core context and shader-storage buffers (optional PyOpenGL;
GLFW on Linux or native WGL on Windows). Both accelerate sparse matrix/vector multiplication;
Newton decisions, Krylov reductions and preconditioning remain on the host.
API references: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html
and https://registry.khronos.org/OpenGL/specs/gl/glspec43.core.pdf .
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import os
import threading

import numpy as np
from scipy import sparse


class AcceleratorUnavailable(RuntimeError):
    """The explicitly requested GPU backend cannot execute on this host."""


def _matrix(matrix):
    if not sparse.issparse(matrix) or len(matrix.shape) != 2:
        raise ValueError("GPU matrix must be a SciPy sparse two-dimensional matrix")
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("GPU Jacobian must contain real FP64 values")
    value = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    value.check_format(full_check=True)
    value.sum_duplicates()
    value.sort_indices()
    if not np.isfinite(value.data).all():
        raise ValueError("GPU matrix contains non-finite values")
    return value


def _vector(value, columns):
    if np.iscomplexobj(value):
        raise ValueError("GPU vector must be real")
    value = np.ascontiguousarray(value, dtype=np.float64)
    if value.ndim != 1 or value.shape[0] != columns:
        raise ValueError(f"GPU vector must have shape ({columns},)")
    if not np.isfinite(value).all():
        raise ValueError("GPU vector contains non-finite values")
    return value


def _check(code, operation):
    if code != 0:
        raise AcceleratorUnavailable(f"OpenCL {operation} failed with status {code}")


class _CL:
    """Exactly typed OpenCL 1.2 entry points; handles and bitfields are not ints."""
    def __init__(self):
        name = "OpenCL.dll" if os.name == "nt" else ctypes.util.find_library("OpenCL")
        if not name:
            raise AcceleratorUnavailable("No system OpenCL ICD loader found")
        try:
            self.lib = ct.WinDLL(name) if os.name == "nt" else ct.CDLL(name)
        except OSError as error:
            raise AcceleratorUnavailable(f"Cannot load OpenCL: {error}") from error
        p, u, z, bits = ct.c_void_p, ct.c_uint, ct.c_size_t, ct.c_uint64
        self._bind("clGetPlatformIDs", ct.c_int, [u, ct.POINTER(p), ct.POINTER(u)])
        self._bind("clGetDeviceIDs", ct.c_int, [p, bits, u, ct.POINTER(p), ct.POINTER(u)])
        self._bind("clGetDeviceInfo", ct.c_int, [p, u, z, p, ct.POINTER(z)])
        self._bind("clGetPlatformInfo", ct.c_int, [p, u, z, p, ct.POINTER(z)])
        self._bind("clCreateContext", p, [p, u, ct.POINTER(p), p, p, ct.POINTER(ct.c_int)])
        self._bind("clCreateCommandQueue", p, [p, p, bits, ct.POINTER(ct.c_int)])
        self._bind("clCreateProgramWithSource", p,
                   [p, u, ct.POINTER(ct.c_char_p), ct.POINTER(z), ct.POINTER(ct.c_int)])
        self._bind("clBuildProgram", ct.c_int, [p, u, ct.POINTER(p), ct.c_char_p, p, p])
        self._bind("clGetProgramBuildInfo", ct.c_int, [p, p, u, z, p, ct.POINTER(z)])
        self._bind("clCreateKernel", p, [p, ct.c_char_p, ct.POINTER(ct.c_int)])
        self._bind("clCreateBuffer", p, [p, bits, z, p, ct.POINTER(ct.c_int)])
        self._bind("clSetKernelArg", ct.c_int, [p, u, z, p])
        for name in ("clEnqueueWriteBuffer", "clEnqueueReadBuffer"):
            self._bind(name, ct.c_int, [p, p, u, z, z, p, u, p, p])
        self._bind("clEnqueueNDRangeKernel", ct.c_int,
                   [p, p, u, ct.POINTER(z), ct.POINTER(z), ct.POINTER(z), u, p, p])
        self._bind("clFinish", ct.c_int, [p])
        for name in ("clReleaseMemObject", "clReleaseKernel", "clReleaseProgram",
                     "clReleaseCommandQueue", "clReleaseContext"):
            self._bind(name, ct.c_int, [p])

    def _bind(self, name, result, arguments):
        function = getattr(self.lib, name)
        function.restype, function.argtypes = result, arguments
        setattr(self, name, function)

    def info(self, function, handle, parameter):
        size = ct.c_size_t()
        _check(function(handle, parameter, 0, None, ct.byref(size)), "information size query")
        data = ct.create_string_buffer(size.value)
        _check(function(handle, parameter, size.value, data, None), "information query")
        return data.raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def devices(self):
        count = ct.c_uint()
        code = self.clGetPlatformIDs(0, None, ct.byref(count))
        if code == -1001 or (code == 0 and count.value == 0):
            return []
        _check(code, "platform enumeration")
        platforms = (ct.c_void_p * count.value)()
        _check(self.clGetPlatformIDs(count, platforms, None), "platform enumeration")
        found = []
        for platform in platforms:
            devices = ct.c_uint()
            code = self.clGetDeviceIDs(platform, 4, 0, None, ct.byref(devices))  # GPU only
            if code == -1:
                continue
            _check(code, "GPU enumeration")
            handles = (ct.c_void_p * devices.value)()
            _check(self.clGetDeviceIDs(platform, 4, devices, handles, None), "GPU enumeration")
            for handle in handles:
                fp64 = ct.c_uint64()
                code = self.clGetDeviceInfo(handle, 0x1032, ct.sizeof(fp64), ct.byref(fp64), None)
                found.append((handle, {
                    "platform": self.info(self.clGetPlatformInfo, platform, 0x0902),
                    "device": self.info(self.clGetDeviceInfo, handle, 0x102B),
                    "fp64": code == 0 and fp64.value != 0,
                    "kind": "GPU",
                }))
        return found


_CL_SOURCE = r"""
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error FP64 extension required
#endif
__kernel void csr_product(__global const ulong *ptr,
                          __global const ulong *col,
                          __global const double *data,
                          __global const double *x,
                          __global double *y,
                          const ulong rows) {
    ulong row = get_global_id(0);
    if (row >= rows) return;
    double total = 0.0;
    for (ulong k = ptr[row]; k < ptr[row + 1]; ++k)
        total += data[k] * x[col[k]];
    y[row] = total;
}
"""


class OpenCLAccelerator:
    def __init__(self, matrix):
        matrix = _matrix(matrix)
        self.shape = matrix.shape
        self._lock = threading.Lock()
        self._closed = False
        self._buffers = []
        self._context = self._queue = self._program = self._kernel = None
        self._api = _CL()
        devices = [(handle, info) for handle, info in self._api.devices() if info["fp64"]]
        if not devices:
            raise AcceleratorUnavailable("No OpenCL GPU with FP64 support found")
        self._device, self.device_info = devices[0]
        api, error = self._api, ct.c_int()
        handles = (ct.c_void_p * 1)(self._device)
        try:
            self._context = api.clCreateContext(None, 1, handles, None, None, ct.byref(error))
            _check(error.value, "context creation")
            self._queue = api.clCreateCommandQueue(self._context, self._device, 0, ct.byref(error))
            _check(error.value, "queue creation")
            source = _CL_SOURCE.encode("ascii")
            strings = (ct.c_char_p * 1)(source)
            lengths = (ct.c_size_t * 1)(len(source))
            self._program = api.clCreateProgramWithSource(self._context, 1, strings, lengths, ct.byref(error))
            _check(error.value, "program creation")
            status = api.clBuildProgram(self._program, 1, handles, b"-cl-std=CL1.2", None, None)
            if status:
                length = ct.c_size_t()
                api.clGetProgramBuildInfo(self._program, self._device, 0x1183, 0, None, ct.byref(length))
                log = ct.create_string_buffer(max(1, length.value))
                api.clGetProgramBuildInfo(self._program, self._device, 0x1183, len(log), log, None)
                raise AcceleratorUnavailable(f"OpenCL kernel build failed ({status}): {log.value.decode(errors='replace')}")
            self._kernel = api.clCreateKernel(self._program, b"csr_product", ct.byref(error))
            _check(error.value, "kernel creation")
            arrays = [np.asarray(matrix.indptr, dtype=np.uint64),
                      np.asarray(matrix.indices, dtype=np.uint64), matrix.data,
                      np.zeros(max(1, matrix.shape[1]), dtype=np.float64),
                      np.zeros(max(1, matrix.shape[0]), dtype=np.float64)]
            for index, array in enumerate(arrays):
                array = np.ascontiguousarray(array)
                if array.size == 0:
                    array = np.zeros(1, dtype=array.dtype)
                flags = (4 if index < 4 else 2) | 32  # READ_ONLY/WRITE_ONLY + COPY_HOST_PTR
                buffer = api.clCreateBuffer(self._context, flags, array.nbytes,
                                            ct.c_void_p(array.ctypes.data), ct.byref(error))
                _check(error.value, "buffer allocation")
                self._buffers.append(buffer)
                argument = ct.c_void_p(buffer)
                _check(api.clSetKernelArg(self._kernel, index, ct.sizeof(argument), ct.byref(argument)), "buffer argument")
            rows = ct.c_uint64(matrix.shape[0])
            _check(api.clSetKernelArg(self._kernel, 5, ct.sizeof(rows), ct.byref(rows)), "row-count argument")
        except Exception:
            self.close()
            raise

    def matvec(self, value):
        value = _vector(value, self.shape[1])
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU accelerator is closed")
            result = np.zeros(self.shape[0], dtype=np.float64)
            if self.shape[0] == 0:
                return result
            api = self._api
            if value.size:
                _check(api.clEnqueueWriteBuffer(self._queue, self._buffers[3], 1, 0,
                                                value.nbytes, ct.c_void_p(value.ctypes.data),
                                                0, None, None), "vector upload")
            work = (ct.c_size_t * 1)(self.shape[0])
            _check(api.clEnqueueNDRangeKernel(self._queue, self._kernel, 1, None,
                                             work, None, 0, None, None), "kernel dispatch")
            _check(api.clEnqueueReadBuffer(self._queue, self._buffers[4], 1, 0,
                                           result.nbytes, ct.c_void_p(result.ctypes.data),
                                           0, None, None), "result download")
            if not np.isfinite(result).all():
                raise ArithmeticError("OpenCL sparse product produced non-finite values")
            return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            api = self._api
            if self._queue:
                api.clFinish(self._queue)
            for buffer in reversed(self._buffers):
                api.clReleaseMemObject(buffer)
            self._buffers.clear()
            for name, release in (("_kernel", api.clReleaseKernel),
                                  ("_program", api.clReleaseProgram),
                                  ("_queue", api.clReleaseCommandQueue),
                                  ("_context", api.clReleaseContext)):
                if getattr(self, name):
                    release(getattr(self, name))
                    setattr(self, name, None)


_GL_SOURCE = r"""#version 430 core
layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer Ptr { uint rowptr[]; };
layout(std430, binding = 1) readonly buffer Col { uint columns[]; };
layout(std430, binding = 2) readonly buffer Data { double values[]; };
layout(std430, binding = 3) readonly buffer X { double x[]; };
layout(std430, binding = 4) writeonly buffer Y { double y[]; };
uniform uint rowCount;
uniform uint rowBase;
void main() {
    uint localRow = gl_GlobalInvocationID.x;
    if (localRow >= rowCount - rowBase) return;
    uint row = rowBase + localRow;
    double total = 0.0;
    for (uint k = rowptr[row]; k < rowptr[row + 1]; ++k)
        total += values[k] * x[columns[k]];
    y[row] = total;
}
"""
_GL_LOCK = threading.Lock()
_GL_USERS = 0
_WGL_PROVIDER = None


class _WGL:
    """Small hidden-context provider using the documented Win32/WGL ABI.

    This creates only its own non-visible 1x1 window. It never activates a window,
    sends input, runs a message loop, or modifies another application's context.
    https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-wglcreatecontext
    https://registry.khronos.org/OpenGL/extensions/ARB/WGL_ARB_create_context.txt
    """
    FALSE, VISIBLE, CONTEXT_VERSION_MAJOR, CONTEXT_VERSION_MINOR = 0, 1, 2, 3
    OPENGL_PROFILE, OPENGL_CORE_PROFILE = 4, 5

    def __init__(self):
        from ctypes import wintypes as wt
        self.user = ct.WinDLL("user32", use_last_error=True)
        self.gdi = ct.WinDLL("gdi32", use_last_error=True)
        self.wgl = ct.WinDLL("opengl32", use_last_error=True)
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        p, integer, uint = ct.c_void_p, ct.c_int, ct.c_uint32

        class WindowClass(ct.Structure):
            _fields_ = [("style", uint), ("procedure", p), ("class_extra", integer),
                        ("window_extra", integer), ("instance", p), ("icon", p),
                        ("cursor", p), ("background", p), ("menu", wt.LPCWSTR),
                        ("name", wt.LPCWSTR)]

        class PixelFormat(ct.Structure):
            _fields_ = [("size", ct.c_uint16), ("version", ct.c_uint16), ("flags", uint),
                        ("pixel_type", ct.c_ubyte), ("color_bits", ct.c_ubyte),
                        ("red_bits", ct.c_ubyte), ("red_shift", ct.c_ubyte),
                        ("green_bits", ct.c_ubyte), ("green_shift", ct.c_ubyte),
                        ("blue_bits", ct.c_ubyte), ("blue_shift", ct.c_ubyte),
                        ("alpha_bits", ct.c_ubyte), ("alpha_shift", ct.c_ubyte),
                        ("accum_bits", ct.c_ubyte), ("accum_red", ct.c_ubyte),
                        ("accum_green", ct.c_ubyte), ("accum_blue", ct.c_ubyte),
                        ("accum_alpha", ct.c_ubyte), ("depth", ct.c_ubyte),
                        ("stencil", ct.c_ubyte), ("aux", ct.c_ubyte),
                        ("layer_type", ct.c_ubyte), ("reserved", ct.c_ubyte),
                        ("layer_mask", uint), ("visible_mask", uint), ("damage_mask", uint)]

        self.WindowClass, self.PixelFormat = WindowClass, PixelFormat
        bindings = [
            (self.kernel, "GetModuleHandleW", p, [wt.LPCWSTR]),
            (self.user, "DefWindowProcW", ct.c_ssize_t, [p, uint, ct.c_size_t, ct.c_ssize_t]),
            (self.user, "RegisterClassW", ct.c_uint16, [ct.POINTER(WindowClass)]),
            (self.user, "UnregisterClassW", integer, [wt.LPCWSTR, p]),
            (self.user, "CreateWindowExW", p,
             [uint, wt.LPCWSTR, wt.LPCWSTR, uint, integer, integer, integer, integer, p, p, p, p]),
            (self.user, "DestroyWindow", integer, [p]),
            (self.user, "GetDC", p, [p]), (self.user, "ReleaseDC", integer, [p, p]),
            (self.gdi, "ChoosePixelFormat", integer, [p, ct.POINTER(PixelFormat)]),
            (self.gdi, "SetPixelFormat", integer, [p, integer, ct.POINTER(PixelFormat)]),
            (self.wgl, "wglCreateContext", p, [p]),
            (self.wgl, "wglDeleteContext", integer, [p]),
            (self.wgl, "wglMakeCurrent", integer, [p, p]),
            (self.wgl, "wglGetCurrentContext", p, []),
            (self.wgl, "wglGetCurrentDC", p, []),
            (self.wgl, "wglGetProcAddress", p, [ct.c_char_p]),
        ]
        for library, name, result, arguments in bindings:
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
        self.instance = self.kernel.GetModuleHandleW(None)
        self.name = f"NRlargeHiddenWGL_{os.getpid()}_{id(self)}"
        self.atom = None
        self.windows = {}

    @staticmethod
    def check(value, operation):
        if not value:
            raise AcceleratorUnavailable(f"WGL {operation} failed: {ct.WinError(ct.get_last_error())}")
        return value

    def init(self):
        if not self.atom:
            cls = self.WindowClass()
            cls.style = 0x20  # CS_OWNDC: stable private device context
            cls.procedure = ct.cast(self.user.DefWindowProcW, ct.c_void_p)
            cls.instance, cls.name = self.instance, self.name
            self.atom = self.check(self.user.RegisterClassW(ct.byref(cls)), "class registration")
        return True

    def terminate(self):
        if self.windows:
            raise RuntimeError("Cannot release WGL provider with live compute contexts")
        if self.atom:
            self.check(self.user.UnregisterClassW(self.name, self.instance), "class release")
            self.atom = None

    def window_hint(self, *_):
        pass  # Native creation below always requests hidden OpenGL 4.3 core.

    def get_current_context(self):
        rc = self.wgl.wglGetCurrentContext()
        if not rc:
            return None
        return self.windows.get(rc, (self.wgl.wglGetCurrentDC(), rc))

    def make_context_current(self, window):
        if window is None and not self.wgl.wglGetCurrentContext():
            return
        dc, rc = (None, None) if window is None else (
            (window["dc"], window["rc"]) if isinstance(window, dict) else window)
        self.check(self.wgl.wglMakeCurrent(dc, rc), "context selection")

    def create_window(self, width, height, title, *_):
        previous = self.get_current_context()
        hwnd = dc = legacy = modern = None
        try:
            # WS_POPUP without WS_VISIBLE: never shown or activated.
            hwnd = self.check(self.user.CreateWindowExW(0, self.name, title, 0x80000000,
                                                       0, 0, width, height, None, None,
                                                       self.instance, None), "hidden window creation")
            dc = self.check(self.user.GetDC(hwnd), "device context creation")
            pixel = self.PixelFormat()
            pixel.size, pixel.version, pixel.flags = ct.sizeof(pixel), 1, 0x4 | 0x20
            pixel.color_bits, pixel.alpha_bits = 24, 8
            number = self.check(self.gdi.ChoosePixelFormat(dc, ct.byref(pixel)), "pixel format selection")
            self.check(self.gdi.SetPixelFormat(dc, number, ct.byref(pixel)), "pixel format setup")
            legacy = self.check(self.wgl.wglCreateContext(dc), "bootstrap context creation")
            self.check(self.wgl.wglMakeCurrent(dc, legacy), "bootstrap context selection")
            address = self.wgl.wglGetProcAddress(b"wglCreateContextAttribsARB")
            if address in (None, 1, 2, 3, ct.c_void_p(-1).value):
                raise AcceleratorUnavailable("WGL_ARB_create_context is unavailable")
            create = ct.WINFUNCTYPE(ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                    ct.POINTER(ct.c_int))(address)
            attributes = (ct.c_int * 7)(0x2091, 4, 0x2092, 3, 0x9126, 1, 0)
            modern = self.check(create(dc, None, attributes), "OpenGL 4.3 core context creation")
            self.check(self.wgl.wglMakeCurrent(dc, modern), "core context selection")
            self.check(self.wgl.wglDeleteContext(legacy), "bootstrap context release")
            legacy = None
            window = {"hwnd": hwnd, "dc": dc, "rc": modern}
            self.windows[modern] = window
            return window
        except Exception:
            self.make_context_current(previous)
            if modern:
                self.wgl.wglDeleteContext(modern)
            if legacy:
                self.wgl.wglDeleteContext(legacy)
            if dc:
                self.user.ReleaseDC(hwnd, dc)
            if hwnd:
                self.user.DestroyWindow(hwnd)
            raise
        finally:
            self.make_context_current(previous)

    def destroy_window(self, window):
        if self.get_current_context() == window:
            self.make_context_current(None)
        self.check(self.wgl.wglDeleteContext(window["rc"]), "core context release")
        self.check(self.user.ReleaseDC(window["hwnd"], window["dc"]), "device context release")
        self.check(self.user.DestroyWindow(window["hwnd"]), "hidden window release")
        self.windows.pop(window["rc"])


def _context_provider():
    try:
        import glfw
        return glfw
    except ImportError as error:
        if os.name != "nt":
            raise AcceleratorUnavailable("OpenGL compute requires optional glfw on this platform") from error
        global _WGL_PROVIDER
        if _WGL_PROVIDER is None:
            _WGL_PROVIDER = _WGL()
        return _WGL_PROVIDER


class OpenGLAccelerator:
    """Context creation/destruction must occur on the application's main thread."""
    def __init__(self, matrix):
        matrix = _matrix(matrix)
        if max(matrix.shape, default=0) >= 2**32 or matrix.nnz >= 2**32:
            raise ValueError("OpenGL CSR shards must fit unsigned 32-bit GPU indices")
        try:
            from OpenGL import GL
            from OpenGL.GL import shaders
            from OpenGL.raw.GL.VERSION.GL_1_5 import (
                glBufferData, glBufferSubData, glGetBufferSubData,
            )
            from OpenGL.raw.GL.VERSION.GL_3_2 import glGetInteger64v
            from OpenGL.raw.GL.VERSION.GL_3_0 import glGetIntegeri_v
        except ImportError as error:
            raise AcceleratorUnavailable("OpenGL compute requires optional PyOpenGL") from error
        glfw = _context_provider()
        if threading.current_thread() is not threading.main_thread():
            raise AcceleratorUnavailable("Create OpenGL accelerators on the process main thread")
        self._glfw, self._gl = glfw, GL
        self._allocate = glBufferData
        self._upload = glBufferSubData
        self._download = glGetBufferSubData
        self.shape = matrix.shape
        self._lock = threading.Lock()
        self._closed = False
        self._window, self._program, self._buffers = None, None, []
        self._registered = False
        global _GL_USERS
        with _GL_LOCK:
            if _GL_USERS == 0 and not glfw.init():
                raise AcceleratorUnavailable("GLFW could not initialize a display/context provider")
            _GL_USERS += 1
            self._registered = True
        previous = glfw.get_current_context()
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            self._window = glfw.create_window(1, 1, "NRlarge compute", None, None)
            if not self._window:
                raise AcceleratorUnavailable("No OpenGL 4.3 core compute context available")
            glfw.make_context_current(self._window)
            major, minor = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION)), int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
            if (major, minor) < (4, 3):
                raise AcceleratorUnavailable("OpenGL 4.3 or newer is required")
            self.device_info = {
                "device": GL.glGetString(GL.GL_RENDERER).decode(errors="replace"),
                "version": GL.glGetString(GL.GL_VERSION).decode(errors="replace"),
                "fp64": True,
            }
            renderer = self.device_info["device"].lower()
            if any(name in renderer for name in ("llvmpipe", "softpipe", "swiftshader",
                                                 "software rasterizer", "gdi generic")):
                raise AcceleratorUnavailable("OpenGL context uses a software renderer, not a GPU")
            self._program = shaders.compileProgram(shaders.compileShader(_GL_SOURCE, GL.GL_COMPUTE_SHADER))
            arrays = [np.asarray(matrix.indptr, dtype=np.uint32),
                      np.asarray(matrix.indices, dtype=np.uint32), matrix.data,
                      np.zeros(max(1, matrix.shape[1]), dtype=np.float64),
                      np.zeros(max(1, matrix.shape[0]), dtype=np.float64)]
            self._buffers = list(np.atleast_1d(GL.glGenBuffers(5)))
            storage_size = ct.c_int64()
            glGetInteger64v(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE, ct.byref(storage_size))
            max_buffer = storage_size.value
            for index, (buffer, array) in enumerate(zip(self._buffers, arrays)):
                array = np.ascontiguousarray(array)
                if not array.size:
                    array = np.zeros(1, dtype=array.dtype)
                if array.nbytes > max_buffer:
                    raise AcceleratorUnavailable(f"OpenGL shard buffer exceeds device block limit ({max_buffer} bytes)")
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, int(buffer))
                self._allocate(GL.GL_SHADER_STORAGE_BUFFER, array.nbytes,
                               ct.c_void_p(array.ctypes.data), GL.GL_DYNAMIC_COPY)
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, index, int(buffer))
            work_groups = ct.c_int()
            glGetIntegeri_v(GL.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, ct.byref(work_groups))
            self._groups = work_groups.value
            self._rows_location = GL.glGetUniformLocation(self._program, "rowCount")
            self._base_location = GL.glGetUniformLocation(self._program, "rowBase")
        except Exception as error:
            glfw.make_context_current(previous)
            self.close()
            if isinstance(error, (AcceleratorUnavailable, ValueError)):
                raise
            raise AcceleratorUnavailable(f"OpenGL compute initialization failed: {error}") from error
        finally:
            glfw.make_context_current(previous)

    def matvec(self, value):
        value = _vector(value, self.shape[1])
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU accelerator is closed")
            result = np.zeros(self.shape[0], dtype=np.float64)
            if not result.size:
                return result
            GL, glfw = self._gl, self._glfw
            previous = glfw.get_current_context()
            glfw.make_context_current(self._window)
            try:
                GL.glUseProgram(self._program)
                for index, buffer in enumerate(self._buffers):
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, index, int(buffer))
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, int(self._buffers[3]))
                if value.size:
                    self._upload(GL.GL_SHADER_STORAGE_BUFFER, 0, value.nbytes,
                                 ct.c_void_p(value.ctypes.data))
                GL.glUniform1ui(self._rows_location, self.shape[0])
                per_dispatch = self._groups * 128
                for base in range(0, self.shape[0], per_dispatch):
                    count = min(per_dispatch, self.shape[0] - base)
                    GL.glUniform1ui(self._base_location, base)
                    GL.glDispatchCompute((count + 127) // 128, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, int(self._buffers[4]))
                self._download(GL.GL_SHADER_STORAGE_BUFFER, 0, result.nbytes,
                               ct.c_void_p(result.ctypes.data))
                if not np.isfinite(result).all():
                    raise ArithmeticError("OpenGL sparse product produced non-finite values")
                return result
            finally:
                glfw.make_context_current(previous)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            glfw, GL = self._glfw, self._gl
            previous = glfw.get_current_context()
            if self._window:
                glfw.make_context_current(self._window)
                if self._buffers:
                    GL.glDeleteBuffers(len(self._buffers), self._buffers)
                if self._program:
                    GL.glDeleteProgram(self._program)
                glfw.make_context_current(None if previous == self._window else previous)
                glfw.destroy_window(self._window)
                self._window = None
            if self._registered:
                global _GL_USERS
                with _GL_LOCK:
                    _GL_USERS -= 1
                    if _GL_USERS == 0:
                        glfw.terminate()
                self._registered = False


def create_accelerator(kind, matrix):
    """Return an explicit GPU backend; unsupported requests raise immediately."""
    if kind == "opencl":
        return OpenCLAccelerator(matrix)
    if kind == "opengl":
        return OpenGLAccelerator(matrix)
    raise ValueError("Accelerator must be 'opencl' or 'opengl'")


def accelerator_info():
    """Probe real devices/contexts; availability means FP64 compute is supported."""
    result = {}
    try:
        devices = [info for _, info in _CL().devices()]
        result["opencl"] = {"available": any(d["fp64"] for d in devices), "devices": devices}
    except (AcceleratorUnavailable, OSError, AttributeError) as error:
        result["opencl"] = {"available": False, "devices": [], "error": str(error)}
    try:
        backend = OpenGLAccelerator(sparse.csr_matrix((0, 0), dtype=np.float64))
        try:
            result["opengl"] = {"available": True, "devices": [backend.device_info]}
        finally:
            backend.close()
    except (AcceleratorUnavailable, OSError) as error:
        result["opengl"] = {"available": False, "devices": [], "error": str(error)}
    return result
