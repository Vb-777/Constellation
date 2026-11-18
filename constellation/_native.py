"""Loader for the C receiver, with a NumPy fallback.

The C library is compiled on first use and cached next to the source. If no
compiler is available the pure-Python path is used instead, so the package
works unbuilt - the C is an optimisation, not a dependency, and `available()`
reports which path is live so a benchmark can never silently time the wrong one.

numpy's complex128 is two contiguous float64 with the real part first, which is
exactly `double _Complex`, so arrays are handed over by pointer. The only thing
required is that they be C-contiguous and of the right dtype, which
`_as` enforces rather than assumes - passing a sliced or transposed array
straight to C would read the wrong memory and produce plausible nonsense.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sysconfig
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "csrc" / "receiver.c"
_LIB = _ROOT / "csrc" / ("receiver" + (sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"))

_lib: ctypes.CDLL | None = None
_reason: str | None = None

_C = ctypes.c_double
_CP = ctypes.POINTER(ctypes.c_double)
_I64 = ctypes.c_int64
_I64P = ctypes.POINTER(ctypes.c_int64)
_U8P = ctypes.POINTER(ctypes.c_uint8)


def _compile() -> Path | None:
    global _reason
    if not _SRC.exists():
        _reason = f"source missing: {_SRC}"
        return None
    if _LIB.exists() and _LIB.stat().st_mtime >= _SRC.stat().st_mtime:
        return _LIB
    cc = os.environ.get("CC") or "cc"
    cmd = [cc, "-O2", "-shared", "-fPIC", "-std=c99",
           "-o", str(_LIB), str(_SRC), "-lm"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        _reason = f"{cc} could not be run: {exc}"
        return None
    if p.returncode != 0:
        _reason = f"{cc} failed: {p.stderr.strip()[:400]}"
        return None
    return _LIB


def load() -> ctypes.CDLL | None:
    """Compile if needed and return the library, or None with a reason recorded."""
    global _lib
    if _lib is not None:
        return _lib
    path = _compile()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # pragma: no cover
        globals()["_reason"] = f"dlopen failed: {exc}"
        return None

    lib.mf_filter.restype = None
    lib.mf_filter.argtypes = [_CP, _I64, _CP, _I64, _CP]
    lib.gardner.restype = _I64
    lib.gardner.argtypes = [_CP, _I64, ctypes.c_int, _C, _C, _CP, _CP, _CP, _I64]
    lib.demap_qam.restype = None
    lib.demap_qam.argtypes = [_CP, _I64, _C, ctypes.c_int, ctypes.c_int, _I64P, _U8P]
    lib.bit_errors.restype = _I64
    lib.bit_errors.argtypes = [_U8P, _U8P, _I64]

    _lib = lib
    return lib


def available() -> bool:
    return load() is not None


def why_unavailable() -> str | None:
    load()
    return _reason


def _as(a: np.ndarray, dtype) -> np.ndarray:
    """Force a C-contiguous array of the required dtype before taking a pointer."""
    return np.ascontiguousarray(a, dtype=dtype)


def _ptr(a: np.ndarray, kind=_CP):
    return a.ctypes.data_as(kind)


# ------------------------------------------------------------------ wrappers
def mf_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    lib = load()
    if lib is None:
        return np.convolve(_as(x, complex), _as(h, float), "full")
    x, h = _as(x, np.complex128), _as(h, np.float64)
    y = np.empty(x.size + h.size - 1, dtype=np.complex128)
    lib.mf_filter(_ptr(x), x.size, _ptr(h), h.size, _ptr(y))
    return y


def gardner(x: np.ndarray, sps: int, kp: float, ki: float, n_sym: int):
    """Returns (symbols, tau, error). Raises if the C library is unavailable."""
    lib = load()
    if lib is None:
        raise RuntimeError(f"C library unavailable: {why_unavailable()}")
    x = _as(x, np.complex128)
    out = np.empty(n_sym, dtype=np.complex128)
    tau = np.empty(n_sym, dtype=np.float64)
    err = np.empty(n_sym, dtype=np.float64)
    n = lib.gardner(_ptr(x), x.size, int(sps), float(kp), float(ki),
                    _ptr(out), _ptr(tau), _ptr(err), int(n_sym))
    return out[:n], tau[:n], err[:n]


def demap_qam(symbols: np.ndarray, alphabet) -> np.ndarray:
    lib = load()
    if lib is None:
        return alphabet.demodulate(symbols)
    s = _as(symbols, np.complex128)
    lut = _as(alphabet.level_to_label, np.int64)
    bits = np.empty(s.size * alphabet.bits_per_symbol, dtype=np.uint8)
    lib.demap_qam(_ptr(s), s.size, float(alphabet.d), int(alphabet.n),
                  int(alphabet.bits_per_dim), _ptr(lut, _I64P), _ptr(bits, _U8P))
    return bits


def bit_errors(a: np.ndarray, b: np.ndarray) -> int:
    lib = load()
    a, b = _as(a, np.uint8), _as(b, np.uint8)
    n = min(a.size, b.size)
    if lib is None:
        return int(np.count_nonzero(a[:n] != b[:n]))
    return int(lib.bit_errors(_ptr(a, _U8P), _ptr(b, _U8P), n))
