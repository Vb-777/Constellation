"""Gray-coded square QAM mapping and hard-decision demapping.

Square QAM factors into two independent Gray-coded PAM arms, so both the
mapper and the slicer work per dimension. That is not just convenient: it is
why the exact BER in `theory.py` is computable at all, and why the demapper
below is genuinely optimal rather than a nearest-neighbour heuristic over the
full 2-D constellation.

Gray coding is the point of the exercise. Adjacent constellation points differ
in exactly one bit, so the overwhelmingly most likely symbol error costs one
bit instead of an average of log2(M)/2.
"""
from __future__ import annotations

import math

import numpy as np

from .theory import _unit_energy_scale, gray_labels, pam_levels

__all__ = ["QAM", "qam", "bits_per_symbol", "random_bits", "ber", "evm_db"]


class QAM:
    """A square M-QAM alphabet with unit mean symbol energy.

    Bit order within a symbol is I bits first, then Q bits, most significant
    first. The normalisation matches `theory.py` exactly, which is what makes
    simulated and analytic curves comparable.
    """

    def __init__(self, M: int) -> None:
        n = int(round(math.sqrt(M)))
        if n * n != M or n < 2 or (n & (n - 1)) != 0:
            raise ValueError(f"M must be a square power of two (4, 16, 64, ...), got {M}")
        self.M = M
        self.n = n
        self.bits_per_dim = int(round(math.log2(n)))
        self.bits_per_symbol = 2 * self.bits_per_dim
        self.d = _unit_energy_scale(n)
        self.levels = self.d * pam_levels(n)

        # label_to_level[g] = amplitude for Gray label g; level_to_label is its inverse.
        labels = gray_labels(n)
        self.level_to_label = labels
        self.label_to_index = np.empty(n, dtype=np.int64)
        self.label_to_index[labels] = np.arange(n)

        i, q = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.points = self.levels[i].ravel() + 1j * self.levels[q].ravel()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"QAM(M={self.M}, bits_per_symbol={self.bits_per_symbol})"

    @property
    def mean_energy(self) -> float:
        return float(np.mean(np.abs(self.points) ** 2))

    @property
    def min_distance(self) -> float:
        """Minimum Euclidean distance between constellation points, = 2d."""
        return 2.0 * self.d

    # ------------------------------------------------------------------ mapping
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Pack bits into complex symbols. len(bits) must be a multiple of bits_per_symbol."""
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if bits.size % self.bits_per_symbol:
            raise ValueError(
                f"{bits.size} bits is not a multiple of {self.bits_per_symbol}"
            )
        groups = bits.reshape(-1, 2, self.bits_per_dim)
        weights = 1 << np.arange(self.bits_per_dim - 1, -1, -1)
        g = groups @ weights                      # (n_sym, 2) Gray labels
        idx = self.label_to_index[g]
        return self.levels[idx[:, 0]] + 1j * self.levels[idx[:, 1]]

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Hard-decide symbols back to bits by per-dimension slicing.

        Slicing each axis to the nearest PAM level is the maximum-likelihood
        decision for square QAM in AWGN, because the noise is independent
        across I and Q and the decision regions are rectangles.
        """
        s = np.asarray(symbols, dtype=complex).ravel()
        iq = np.stack([s.real, s.imag], axis=1)
        # Nearest level index without a search: the levels are uniform.
        idx = np.rint((iq / self.d + (self.n - 1)) / 2.0)
        np.clip(idx, 0, self.n - 1, out=idx)
        g = self.level_to_label[idx.astype(np.int64)]       # (n_sym, 2)
        shifts = np.arange(self.bits_per_dim - 1, -1, -1)
        bits = (g[:, :, None] >> shifts) & 1
        return bits.reshape(-1).astype(np.uint8)

    def nearest_points(self, symbols: np.ndarray) -> np.ndarray:
        """Snap symbols to the constellation. Used by decision-directed recovery."""
        s = np.asarray(symbols, dtype=complex)
        idx_i = np.clip(np.rint((s.real / self.d + (self.n - 1)) / 2.0), 0, self.n - 1)
        idx_q = np.clip(np.rint((s.imag / self.d + (self.n - 1)) / 2.0), 0, self.n - 1)
        return (self.levels[idx_i.astype(np.int64)]
                + 1j * self.levels[idx_q.astype(np.int64)])


_CACHE: dict[int, QAM] = {}


def qam(M: int) -> QAM:
    """Cached alphabet, so sweeps do not rebuild the tables for every SNR point."""
    if M not in _CACHE:
        _CACHE[M] = QAM(M)
    return _CACHE[M]


def bits_per_symbol(M: int) -> int:
    return qam(M).bits_per_symbol


def random_bits(n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit error rate over the overlapping prefix of the two streams."""
    n = min(tx_bits.size, rx_bits.size)
    if n == 0:
        raise ValueError("no bits to compare")
    return float(np.count_nonzero(tx_bits[:n] != rx_bits[:n])) / n


def evm_db(rx: np.ndarray, ref: np.ndarray) -> float:
    """Error-vector magnitude in dB relative to reference RMS power.

    EVM is what a spectrum analyser reports and what a modem team argues about,
    so it is worth having alongside BER: it degrades smoothly where BER is
    still exactly zero, which makes it the more useful number for diagnosing an
    impairment before it costs any bits.
    """
    rx, ref = np.asarray(rx, dtype=complex), np.asarray(ref, dtype=complex)
    n = min(rx.size, ref.size)
    err = float(np.mean(np.abs(rx[:n] - ref[:n]) ** 2))
    ref_pow = float(np.mean(np.abs(ref[:n]) ** 2))
    if err == 0.0:
        return -np.inf
    return 10.0 * math.log10(err / ref_pow)
