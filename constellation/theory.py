"""Analytic error probabilities. These are the yardstick everything is measured against.

The AWGN result is computed exactly, not approximated: for square QAM the
in-phase and quadrature arms are independent Gray-coded PAM, so the bit error
probability is obtained by integrating the Gaussian over every decision region
and counting the Hamming distance between the transmitted and decided labels.
No high-SNR approximation and no union bound.

A useful self-check falls out of this: for M = 4 the enumeration collapses to
the textbook Q(sqrt(2 Eb/N0)), and `test_theory.py` asserts that it does.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

__all__ = [
    "qfunc",
    "db_to_linear",
    "awgn_ber_qpsk",
    "awgn_ber_square_qam",
    "awgn_ser_square_qam",
    "rayleigh_ber_qpsk",
    "gray_labels",
    "pam_levels",
]


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail Q(x) = P(N(0,1) > x), via erfc so the far tail stays accurate.

    Computing it as ``1 - norm.cdf(x)`` loses all significance past x ~ 8;
    erfc keeps working to x ~ 26 before underflowing to zero.
    """
    return 0.5 * erfc(np.asarray(x, dtype=float) / math.sqrt(2.0))


def db_to_linear(db: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def gray_labels(n: int) -> np.ndarray:
    """Binary-reflected Gray code for level indices 0..n-1."""
    i = np.arange(n)
    return i ^ (i >> 1)


def pam_levels(n: int) -> np.ndarray:
    """Symmetric PAM amplitudes for n levels, before energy normalisation: -n+1 .. n-1."""
    return 2.0 * np.arange(n) - (n - 1)


def _region_probs(levels: np.ndarray, edges: np.ndarray, sigma: float) -> np.ndarray:
    """P(decide region j | level i sent), evaluated without cancelling in the tails.

    The obvious implementation, ``cdf(v) - cdf(u)``, silently loses the whole
    answer at high SNR: Phi(9) differs from 1.0 by 1e-19, which is far below
    float64 resolution at 1.0, so the subtraction returns exactly zero. The
    error probability for the level *below* the boundary vanishes while the one
    above it survives as a left tail, and the total comes out a factor of two
    low. That factor of two is not visible at low SNR, which is exactly what
    makes it dangerous.

    So each region is integrated from whichever end does not cancel: a region
    entirely to the right of the mean is a difference of survival functions, a
    region to the left is a difference of CDFs, and a region straddling the
    mean is safe either way.
    """
    u = (edges[None, :-1] - levels[:, None]) / sigma   # lower edge, standardised
    v = (edges[None, 1:] - levels[:, None]) / sigma    # upper edge
    right = u >= 0.0
    return np.where(right,
                    norm.sf(u) - norm.sf(np.where(right, v, 0.0)),
                    norm.cdf(v) - norm.cdf(np.where(right, 0.0, u)))


def _unit_energy_scale(n_per_dim: int) -> float:
    """Scale d such that a square QAM built from these levels has unit mean energy.

    Mean symbol energy is E[(d*a_I)^2 + (d*a_Q)^2] = 2 d^2 E[a^2]; setting that
    to 1 fixes d. Every modulator and every theory curve in this project uses
    the same normalisation, which is what makes the two comparable at all.
    """
    a = pam_levels(n_per_dim)
    return 1.0 / math.sqrt(2.0 * float(np.mean(a**2)))


def awgn_ber_qpsk(ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """Gray-coded QPSK: BER = Q(sqrt(2 Eb/N0)), exact."""
    return qfunc(np.sqrt(2.0 * db_to_linear(ebn0_db)))


def awgn_ber_square_qam(M: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Exact Gray-coded BER for square M-QAM over AWGN with ML detection.

    Integrates the per-dimension Gaussian over all sqrt(M) decision regions and
    weights by Hamming distance. Exact for every SNR, not just high SNR.
    """
    n = int(round(math.sqrt(M)))
    if n * n != M or n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"M must be a square power of two (4, 16, 64, ...), got {M}")

    bits_per_dim = int(round(math.log2(n)))
    bits_per_sym = 2 * bits_per_dim
    d = _unit_energy_scale(n)
    levels = d * pam_levels(n)
    labels = gray_labels(n)

    # Hamming distance between every pair of Gray labels.
    xor = labels[:, None] ^ labels[None, :]
    hamming = np.zeros_like(xor, dtype=float)
    for b in range(bits_per_dim):
        hamming += (xor >> b) & 1

    # Decision boundaries midway between adjacent levels.
    edges = np.concatenate(([-np.inf], 0.5 * (levels[:-1] + levels[1:]), [np.inf]))

    ebn0 = np.atleast_1d(db_to_linear(ebn0_db))
    n0 = 1.0 / (bits_per_sym * ebn0)          # Es = 1 by construction
    sigma = np.sqrt(n0 / 2.0)                  # noise std per dimension

    out = np.empty_like(ebn0)
    for k, s in enumerate(sigma):
        p = _region_probs(levels, edges, s)    # P(decide j | sent i)
        # Equiprobable levels; divide by bits per dimension to get a bit rate.
        out[k] = float(np.sum(p * hamming)) / (n * bits_per_dim)

    return out if np.ndim(ebn0_db) else out.reshape(())[()]


def awgn_ser_square_qam(M: int, ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """Exact symbol error rate for square M-QAM: 1 - (1 - P_dim)^2."""
    n = int(round(math.sqrt(M)))
    bits_per_sym = 2 * int(round(math.log2(n)))
    d = _unit_energy_scale(n)
    levels = d * pam_levels(n)
    edges = np.concatenate(([-np.inf], 0.5 * (levels[:-1] + levels[1:]), [np.inf]))

    ebn0 = np.atleast_1d(db_to_linear(ebn0_db))
    sigma = np.sqrt(1.0 / (bits_per_sym * ebn0) / 2.0)

    out = np.empty_like(ebn0)
    for k, s in enumerate(sigma):
        p = _region_probs(levels, edges, s)
        correct_dim = float(np.mean(np.diag(p)))
        out[k] = 1.0 - correct_dim**2
    return out if np.ndim(ebn0_db) else out.reshape(())[()]


def rayleigh_ber_qpsk(ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """Gray QPSK over flat Rayleigh fading with coherent detection and perfect CSI.

    BER = (1/2)(1 - sqrt(g/(1+g))), g = mean Eb/N0.

    The whole point of this curve is its slope: averaging Q(sqrt(2g|h|^2)) over
    an exponential |h|^2 turns exponential decay into decay like 1/(4g). One
    deep fade dominates the average, so 1e-5 needs roughly 40 dB rather than
    10 dB. That gap is the reason diversity and coding exist.
    """
    g = db_to_linear(ebn0_db)
    return 0.5 * (1.0 - np.sqrt(g / (1.0 + g)))
