"""GPU-accelerated CA dynamics via HIP kernel (AMD ROCm).

Loads the compiled HIP shared library and provides a Python interface
matching the CPU evolve_and_interpret API.  Falls back gracefully
when the GPU library is not available.

Library preference order (first found wins):
  1. libwheeler_ca_v2.so  — variable grid size, exposes stability_threshold
  2. libwheeler_ca.so     — v1, fixed 64×64 grid
"""

import ctypes
import os
import numpy as np

from .dynamics import apply_ca_dynamics  # CPU fallback for get_cell_roles

_LIB_DIR = os.path.join(os.path.dirname(__file__), "gpu")
_LIB_V2_PATH = os.path.join(_LIB_DIR, "libwheeler_ca_v2.so")
_LIB_V1_PATH = os.path.join(_LIB_DIR, "libwheeler_ca.so")

_lib = None
_lib_version = None  # 1 or 2


def _load_lib():
    """Try to load the HIP shared library (v2 preferred, v1 fallback)."""
    global _lib, _lib_version
    if _lib is not None:
        return _lib

    # Try v2 first
    if os.path.exists(_LIB_V2_PATH):
        try:
            lib = ctypes.CDLL(_LIB_V2_PATH)

            # int ca_evolve_batch_v2(float*, float*, int*, int*, int, int, int, float)
            lib.ca_evolve_batch_v2.argtypes = [
                ctypes.POINTER(ctypes.c_float),   # frames_in
                ctypes.POINTER(ctypes.c_float),   # frames_out
                ctypes.POINTER(ctypes.c_int),     # ticks_out
                ctypes.POINTER(ctypes.c_int),     # states_out
                ctypes.c_int,                     # batch_size
                ctypes.c_int,                     # grid_w
                ctypes.c_int,                     # max_iters
                ctypes.c_float,                   # stability_threshold
            ]
            lib.ca_evolve_batch_v2.restype = ctypes.c_int

            # int ca_evolve_single_v2(float*, float*, int*, int*, int, int, float)
            lib.ca_evolve_single_v2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,                     # grid_w
                ctypes.c_int,                     # max_iters
                ctypes.c_float,                   # stability_threshold
            ]
            lib.ca_evolve_single_v2.restype = ctypes.c_int

            # size_t ca_query_vram(int batch_size, int grid_w)
            lib.ca_query_vram.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.ca_query_vram.restype = ctypes.c_size_t

            _lib = lib
            _lib_version = 2
            return _lib
        except OSError as e:
            print(f"Warning: could not load GPU library v2: {e}")

    # Fall back to v1
    if os.path.exists(_LIB_V1_PATH):
        try:
            lib = ctypes.CDLL(_LIB_V1_PATH)

            # int ca_evolve_batch(float*, float*, int*, int*, int, int)
            lib.ca_evolve_batch.argtypes = [
                ctypes.POINTER(ctypes.c_float),   # frames_in
                ctypes.POINTER(ctypes.c_float),   # frames_out
                ctypes.POINTER(ctypes.c_int),     # ticks_out
                ctypes.POINTER(ctypes.c_int),     # states_out
                ctypes.c_int,                     # batch_size
                ctypes.c_int,                     # max_iters
            ]
            lib.ca_evolve_batch.restype = ctypes.c_int

            # int ca_evolve_single(float*, float*, int*, int*, int)
            lib.ca_evolve_single.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            lib.ca_evolve_single.restype = ctypes.c_int

            _lib = lib
            _lib_version = 1
            return _lib
        except OSError as e:
            print(f"Warning: could not load GPU library v1: {e}")

    return None


def gpu_available() -> bool:
    """Check if the GPU backend is ready."""
    return _load_lib() is not None


def gpu_version() -> int | None:
    """Return the loaded kernel version (1 or 2), or None if not loaded."""
    _load_lib()
    return _lib_version


def gpu_evolve_single(
    frame: np.ndarray,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
    grid_w: int = 64,
) -> dict:
    """Evolve a single frame on GPU.

    Returns dict with same keys as evolve_and_interpret:
      - state, attractor, convergence_ticks, metadata
      - history is NOT available (GPU doesn't store per-tick frames)

    Args:
        frame: 2-D float32 array (grid_w × grid_w)
        max_iters: max CA iterations
        stability_threshold: convergence threshold (v2 only; v1 uses compiled-in value)
        grid_w: grid width — any value for v2, must be 64 for v1
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("GPU library not available. Build with: cd wheeler_memory/gpu && make")

    cells = grid_w * grid_w
    frame_in = np.ascontiguousarray(frame.flatten(), dtype=np.float32)
    frame_out = np.zeros(cells, dtype=np.float32)
    ticks = ctypes.c_int(0)
    state = ctypes.c_int(0)

    if _lib_version == 2:
        ret = lib.ca_evolve_single_v2(
            frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(ticks),
            ctypes.byref(state),
            grid_w,
            max_iters,
            ctypes.c_float(stability_threshold),
        )
    else:
        ret = lib.ca_evolve_single(
            frame_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            frame_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(ticks),
            ctypes.byref(state),
            max_iters,
        )

    if ret != 0:
        raise RuntimeError("GPU kernel execution failed")

    state_names = {0: "CONVERGED", 1: "OSCILLATING", 2: "CHAOTIC"}

    return {
        "state": state_names.get(state.value, "CHAOTIC"),
        "attractor": frame_out.reshape(grid_w, grid_w),
        "convergence_ticks": ticks.value,
        "history": [],  # GPU path doesn't store history
        "metadata": {"backend": f"gpu_v{_lib_version}", "grid_w": grid_w},
    }


def gpu_evolve_batch(
    frames: list[np.ndarray],
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
    grid_w: int = 64,
) -> list[dict]:
    """Evolve a batch of frames on GPU in parallel.

    Args:
        frames: list of N (grid_w × grid_w) numpy arrays
        max_iters: max CA iterations
        stability_threshold: convergence threshold (v2 only; v1 ignores it)
        grid_w: grid width — any value for v2, must be 64 for v1

    Returns:
        list of N result dicts (same format as evolve_and_interpret,
        but without history)
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("GPU library not available. Build with: cd wheeler_memory/gpu && make")

    batch_size = len(frames)
    if batch_size == 0:
        return []

    cells = grid_w * grid_w
    flat_in = np.zeros(batch_size * cells, dtype=np.float32)
    for i, f in enumerate(frames):
        flat_in[i * cells : (i + 1) * cells] = f.flatten().astype(np.float32)

    flat_out = np.zeros_like(flat_in)
    ticks_out = np.zeros(batch_size, dtype=np.int32)
    states_out = np.zeros(batch_size, dtype=np.int32)

    if _lib_version == 2:
        ret = lib.ca_evolve_batch_v2(
            flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ticks_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            batch_size,
            grid_w,
            max_iters,
            ctypes.c_float(stability_threshold),
        )
    else:
        ret = lib.ca_evolve_batch(
            flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flat_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ticks_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            states_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            batch_size,
            max_iters,
        )

    if ret != 0:
        raise RuntimeError("GPU kernel execution failed")

    state_names = {0: "CONVERGED", 1: "OSCILLATING", 2: "CHAOTIC"}

    results = []
    for i in range(batch_size):
        results.append({
            "state": state_names.get(int(states_out[i]), "CHAOTIC"),
            "attractor": flat_out[i * cells : (i + 1) * cells].reshape(grid_w, grid_w).copy(),
            "convergence_ticks": int(ticks_out[i]),
            "history": [],
            "metadata": {"backend": f"gpu_v{_lib_version}", "grid_w": grid_w},
        })

    return results


def gpu_query_vram(batch_size: int, grid_w: int = 64) -> int | None:
    """Estimate VRAM bytes needed for a given batch + grid config (v2 only).

    Returns None if v1 is loaded (v1 doesn't expose this function).
    """
    lib = _load_lib()
    if lib is None or _lib_version != 2:
        return None
    return lib.ca_query_vram(batch_size, grid_w)
