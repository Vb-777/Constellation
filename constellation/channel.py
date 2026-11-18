"""Channel impairments: noise, flat fading, multipath, and front-end offsets.

Noise scaling is the one thing in a link simulator that is easy to get wrong
and invisible when wrong - a curve 1 dB off looks perfectly plausible. The
convention here is stated once and used everywhere:

    complex noise variance per sample  sigma^2 = 1 / (bits_per_symbol * Eb/N0)

with symbols normalised to unit mean energy. Because the receive filter has
unit energy (see `pulse.py`), that same sigma^2 reaches the slicer whatever the
oversampling factor, so a BER measured at 4 samples per symbol must equal one
measured at 8. `test_channel.py` asserts exactly that, which is the cheapest
available guard against a silent dB error.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["noise_sigma", "awgn", "rayleigh_flat", "multipath", "phase_offset",
           "frequency_offset", "fractional_delay", "EPA_TAPS", "exponential_taps"]


def _generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def noise_sigma(ebn0_db: float, bits_per_symbol: int) -> float:
    """Std of the complex noise per sample for unit-energy symbols."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return math.sqrt(1.0 / (bits_per_symbol * ebn0))


def awgn(x: np.ndarray, ebn0_db: float, bits_per_symbol: int,
         rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Add complex AWGN at the stated Eb/N0.

    The variance is split evenly between the real and imaginary parts, so each
    dimension carries sigma^2/2 - which is the N0/2 that the analytic BER in
    `theory.py` assumes.
    """
    g = _generator(rng)
    x = np.asarray(x, dtype=complex)
    sigma = noise_sigma(ebn0_db, bits_per_symbol)
    n = (g.standard_normal(x.shape) + 1j * g.standard_normal(x.shape)) * (sigma / math.sqrt(2.0))
    return x + n


def rayleigh_flat(x: np.ndarray, rng: np.random.Generator | int | None = None,
                  block: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Flat Rayleigh fading, unit mean power. Returns (faded signal, gains).

    `block` is the coherence length in samples: 1 is fully interleaved fading
    (every symbol an independent draw), large values are a slowly fading
    channel. It matters a great deal - the same average SNR gives very
    different error statistics depending on whether fades are spread out or
    arrive in bursts, and only the interleaved case matches the textbook
    average-BER curve in `theory.rayleigh_ber_qpsk`.
    """
    g = _generator(rng)
    x = np.asarray(x, dtype=complex)
    n_blocks = -(-x.size // block)
    h = (g.standard_normal(n_blocks) + 1j * g.standard_normal(n_blocks)) / math.sqrt(2.0)
    h = np.repeat(h, block)[: x.size]
    return x * h, h


def multipath(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve with a tapped delay line, keeping the input length.

    The tail past the end of the block is dropped, which is the honest model
    for a continuously transmitting link: the energy that spills past a block
    boundary lands on the next block, not nowhere. OFDM's cyclic prefix is what
    makes that spill harmless, and `ofdm.py` demonstrates it.
    """
    x = np.asarray(x, dtype=complex)
    taps = np.asarray(taps, dtype=complex)
    return np.convolve(x, taps)[: x.size]


def exponential_taps(n: int, decay_db: float = 3.0,
                     rng: np.random.Generator | int | None = None) -> np.ndarray:
    """n-tap channel with an exponentially decaying power profile, unit total power.

    Taps are complex Gaussian, so each realisation is a frequency-selective
    Rayleigh channel. Normalising total power to one keeps the SNR definition
    intact, so a BER degradation can only come from selectivity, not from the
    channel quietly swallowing or adding energy.
    """
    g = _generator(rng)
    power = 10.0 ** (-decay_db * np.arange(n) / 10.0)
    power /= power.sum()
    h = (g.standard_normal(n) + 1j * g.standard_normal(n)) / math.sqrt(2.0)
    return h * np.sqrt(power)


#: A fixed 3-tap frequency-selective channel, used where a repeatable
#: multipath profile matters more than an ensemble average.
EPA_TAPS = np.array([0.8, 0.0, 0.5 - 0.3j, 0.0, 0.2j], dtype=complex)
EPA_TAPS = EPA_TAPS / np.sqrt(np.sum(np.abs(EPA_TAPS) ** 2))


def phase_offset(x: np.ndarray, radians: float) -> np.ndarray:
    """Constant phase rotation - an unlocked carrier."""
    return np.asarray(x, dtype=complex) * np.exp(1j * radians)


def frequency_offset(x: np.ndarray, normalised_hz: float,
                     phase0: float = 0.0) -> np.ndarray:
    """Rotate progressively. `normalised_hz` is cycles per sample.

    A 1 ppm crystal error at 2 GHz is 2 kHz, which at a 1 Msps sample rate is
    2e-3 cycles per sample - enough to spin the constellation through a full
    turn every 500 samples. Carrier recovery is not optional.
    """
    x = np.asarray(x, dtype=complex)
    n = np.arange(x.size)
    return x * np.exp(1j * (2.0 * np.pi * normalised_hz * n + phase0))


def fractional_delay(x: np.ndarray, delay: float, n_taps: int = 21) -> np.ndarray:
    """Delay by a non-integer number of samples with a windowed-sinc interpolator.

    This is how a sampling-phase error is injected: the receiver's ADC does not
    happen to sample at the pulse peak, and the timing loop has to find it.
    """
    x = np.asarray(x, dtype=complex)
    if n_taps % 2 == 0:
        n_taps += 1
    half = n_taps // 2
    k = np.arange(-half, half + 1)
    h = np.sinc(k - delay) * np.hamming(n_taps)
    h /= h.sum()
    return np.convolve(x, h)[half: half + x.size]
