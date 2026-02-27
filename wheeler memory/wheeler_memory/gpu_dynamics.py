"""GPU-accelerated CA dynamics via HIP kernel (AMD ROCm).

Loads the compiled libwheeler_ca.so and provides a Python interface
matching the CPU evolve_and_interpret API.  Falls back gracefully
when the GPU library is not available.
"""

import ctypes
import os
import numpy as np

from .dynamics import apply_ca_dynamics  # CPU fallback for get_cell_roles

_LIB_NAME = "libwheeler_ca.so"
_LIB_DIR = os.path.join(os.path.dirname(__file__), "gpu")
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)

_lib = None


def _load_lib():
    """Try to load the HIP shared library."""
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        return None
    try:
        _lib = ctypes.CDLL(_LIB_PATH)

        # int ca_evolve_batch(float*, float*, int*, int*, int, int)
        _lib.ca_evolve_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # frames_in
            ctypes.POINTER(ctypes.c_float),   # frames_out
            ctypes.POINTER(ctypes.c_int),     # ticks_out
            ctypes.POINTER(ctypes.c_int),     # states_out
            ctypes.c_int,                     # batch_size
            ctypes.c_int,                     # max_iters
        ]
        _lib.ca_evolve_batch.restype = ctypes.c_int

        # int ca_evolve_single(float*, float*, int*, int*, int)
        _lib.ca_evolve_single.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        _lib.ca_evolve_single.restype = ctypes.c_int

        return _lib
    except OSError as e:
        print(f"Warning: could not load GPU library: {e}")
        return None


def gpu_available() -> bool:
    """Check if the GPU backend is ready."""
    return _load_lib() is not None


def gpu_evolve_single(
    frame: np.ndarray,
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
) -> dict:
    """Evolve a single frame on GPU.

    Returns dict with same keys as evolve_and_interpret:
      - state, attractor, convergence_ticks, metadata
      - history is NOT available (GPU doesn't store per-tick frames)

    Note: stability_threshold is accepted for API compatibility but
    ignored — the GPU kernel uses a compiled-in threshold value.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("GPU library not available. Build with: cd wheeler_memory/gpu && make")

    frame_in = np.ascontiguousarray(frame.flatten(), dtype=np.float32)
    frame_out = np.zeros(64 * 64, dtype=np.float32)
    ticks = ctypes.c_int(0)
    state = ctypes.c_int(0)

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
        "attractor": frame_out.reshape(64, 64),
        "convergence_ticks": ticks.value,
        "history": [],  # GPU path doesn't store history
        "metadata": {"backend": "gpu"},
    }


def gpu_evolve_batch(
    frames: list[np.ndarray],
    max_iters: int = 1000,
    stability_threshold: float = 1e-4,
) -> list[dict]:
    """Evolve a batch of frames on GPU in parallel.

    Args:
        frames: list of N 64×64 numpy arrays
        max_iters: max CA iterations
        stability_threshold: accepted for API compatibility (ignored by GPU)

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

    # Pack all frames into a contiguous float32 buffer
    flat_in = np.zeros(batch_size * 64 * 64, dtype=np.float32)
    for i, f in enumerate(frames):
        flat_in[i * 4096 : (i + 1) * 4096] = f.flatten().astype(np.float32)

    flat_out = np.zeros_like(flat_in)
    ticks_out = np.zeros(batch_size, dtype=np.int32)
    states_out = np.zeros(batch_size, dtype=np.int32)

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
            "attractor": flat_out[i * 4096 : (i + 1) * 4096].reshape(64, 64).copy(),
            "convergence_ticks": int(ticks_out[i]),
            "history": [],
            "metadata": {"backend": "gpu"},
        })

    return results
