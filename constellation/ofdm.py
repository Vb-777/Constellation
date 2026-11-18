"""OFDM: cyclic prefix, pilot-based channel estimation, zero-forcing equalisation.

The reason OFDM is in every modern standard is one structural fact: prepend a
copy of the tail of each block long enough to cover the channel's delay spread,
and linear convolution with the channel becomes *circular* convolution. A
circular convolution is diagonal in the DFT basis, so a frequency-selective
channel that would need an equaliser with many taps becomes N independent
complex multiplications - one per subcarrier - that a single divide undoes.

This module demonstrates that rather than asserting it, and the demonstration
is quantitative. With the prefix longer than the channel there is no residual
ISI at all, so the bit error rate must equal the average over subcarriers of
the AWGN error rate at each subcarrier's own SNR:

    BER = (1/D) * sum_k  BER_awgn( |H_k|^2 / sigma^2 )

That prediction is exact, it accounts for the frequency selectivity rather than
averaging it away, and it leaves cyclic-prefix sufficiency as the only possible
source of disagreement. `test_ofdm.py` checks it, and shortening the prefix
below the delay spread breaks it in the way the theory says it should.

Energy bookkeeping, which is where OFDM simulations usually go wrong: the
prefix is transmitted energy that carries no new information and the pilots
carry no data, so with N subcarriers, N_cp of prefix and D data subcarriers the
useful Eb/N0 is below the channel Eb/N0 by exactly

    10 log10( (N + N_cp) / D )   dB

- 1.55 dB for the defaults here. Overlooking that term is what produces an
OFDM curve that appears to beat single-carrier QAM, which is impossible.
"""
from __future__ import annotations

import math

import numpy as np

from .modulation import qam
from .theory import awgn_ber_square_qam

__all__ = ["OFDMConfig", "modulate", "demodulate", "estimate_channel",
           "overhead_db", "per_subcarrier_ber", "simulate", "OFDMResult"]


class OFDMConfig:
    """Subcarrier layout. Defaults are a small, LTE-flavoured configuration."""

    __slots__ = ("n_sc", "n_cp", "pilot_spacing", "M")

    def __init__(self, n_sc: int = 64, n_cp: int = 16,
                 pilot_spacing: int = 8, M: int = 4) -> None:
        if n_sc < 8 or (n_sc & (n_sc - 1)):
            raise ValueError(f"n_sc must be a power of two >= 8, got {n_sc}")
        if not 0 <= n_cp < n_sc:
            raise ValueError(f"n_cp must be in [0, n_sc), got {n_cp}")
        if pilot_spacing < 2 or n_sc % pilot_spacing:
            raise ValueError(f"pilot_spacing must divide n_sc and be >= 2, got {pilot_spacing}")
        self.n_sc, self.n_cp, self.pilot_spacing, self.M = n_sc, n_cp, pilot_spacing, M

    def __repr__(self) -> str:  # pragma: no cover
        return (f"OFDMConfig(n_sc={self.n_sc}, n_cp={self.n_cp}, "
                f"pilot_spacing={self.pilot_spacing}, M={self.M})")

    @property
    def pilot_idx(self) -> np.ndarray:
        return np.arange(0, self.n_sc, self.pilot_spacing)

    @property
    def data_idx(self) -> np.ndarray:
        mask = np.ones(self.n_sc, dtype=bool)
        mask[self.pilot_idx] = False
        return np.flatnonzero(mask)

    @property
    def n_data(self) -> int:
        return self.data_idx.size

    @property
    def bits_per_block(self) -> int:
        return self.n_data * qam(self.M).bits_per_symbol

    @property
    def samples_per_block(self) -> int:
        return self.n_sc + self.n_cp

    @property
    def pilots(self) -> np.ndarray:
        """Unit-modulus pilot values. A fixed pseudo-random BPSK pattern.

        Constant modulus matters: least-squares estimation divides the received
        pilot by the known one, so a low-amplitude pilot would hand that
        subcarrier's estimate a correspondingly amplified noise term.
        """
        rng = np.random.default_rng(0xBEEF)
        return (2.0 * rng.integers(0, 2, size=self.pilot_idx.size) - 1.0).astype(complex)


def overhead_db(cfg: OFDMConfig) -> float:
    """Eb/N0 penalty from the cyclic prefix and the pilot subcarriers, in dB."""
    return 10.0 * math.log10(cfg.samples_per_block / cfg.n_data)


def noise_sigma2(cfg: OFDMConfig, ebn0_db: float) -> float:
    """Time-domain complex noise variance for a stated channel Eb/N0.

    Unit mean energy per time-domain sample, so energy per block is
    N + N_cp and the bits per block are D * log2(M).
    """
    m = qam(cfg.M).bits_per_symbol
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    eb = cfg.samples_per_block / (cfg.n_data * m)
    return eb / ebn0


def modulate(bits: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """Bits -> time-domain samples, with pilots inserted and a prefix prepended."""
    a = qam(cfg.M)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size % cfg.bits_per_block:
        raise ValueError(f"{bits.size} bits is not a multiple of "
                         f"{cfg.bits_per_block} bits per OFDM block")
    n_blocks = bits.size // cfg.bits_per_block
    data = a.modulate(bits).reshape(n_blocks, cfg.n_data)

    grid = np.zeros((n_blocks, cfg.n_sc), dtype=complex)
    grid[:, cfg.data_idx] = data
    grid[:, cfg.pilot_idx] = cfg.pilots

    # Unitary IFFT, so unit energy per subcarrier becomes unit energy per sample.
    x = np.fft.ifft(grid, axis=1) * math.sqrt(cfg.n_sc)
    with_cp = np.concatenate([x[:, cfg.n_sc - cfg.n_cp:], x], axis=1) if cfg.n_cp else x
    return with_cp.ravel()


def estimate_channel(grid: np.ndarray, cfg: OFDMConfig,
                     method: str = "dft", n_taps: int | None = None) -> np.ndarray:
    """Channel estimate from the pilots, by one of two reconstructions.

    Both start from the same least-squares pilot observations H_p = Y_p / X_p.
    They differ in how the remaining subcarriers are filled in, and the
    difference is not a detail - it decides whether the estimator has an error
    floor.

    ``"linear"`` interpolates magnitude and unwrapped phase between pilots.
    Magnitude and phase rather than real and imaginary parts, because the
    straight line between two pilots whose phases differ by nearly pi passes
    close to the origin, which would collapse the estimated gain and make the
    equaliser amplify that subcarrier's noise enormously. Even done carefully it
    leaves a bias: pilot spacing of 8 satisfies the sampling theorem for this
    channel (spacing must be below N/L), but a straight line is not the
    reconstruction filter that sampling theorem calls for. The channel response
    between pilots is curved, linear interpolation cuts the corner, and the
    residual does not shrink with SNR - it is a modelling error, not noise. That
    is a measured error floor near 6e-2, and it is there at any SNR.

    ``"dft"`` instead solves for the channel's time-domain taps. With P
    uniformly spaced pilots, H_p = A h where A is a DFT submatrix, so least
    squares recovers up to P taps and an FFT gives every subcarrier. This is
    exact whenever the channel is no longer than the number of pilots - no
    interpolation and no floor - and it is what a real receiver does. The
    assumption it rests on is a bounded delay spread, which is the same
    assumption that sized the cyclic prefix, so it costs nothing new.
    """
    if method not in ("dft", "linear"):
        raise ValueError(f"method must be 'dft' or 'linear', got {method!r}")

    h_p = grid[:, cfg.pilot_idx] / cfg.pilots[None, :]

    if method == "linear":
        k_p = cfg.pilot_idx.astype(float)
        k_all = np.arange(cfg.n_sc, dtype=float)
        mag = np.abs(h_p)
        phase = np.unwrap(np.angle(h_p), axis=1)
        out = np.empty((grid.shape[0], cfg.n_sc), dtype=complex)
        for b in range(grid.shape[0]):
            out[b] = (np.interp(k_all, k_p, mag[b])
                      * np.exp(1j * np.interp(k_all, k_p, phase[b])))
        return out

    # Transform domain: solve H_p = A h for the taps, then transform back.
    # L is capped by the pilot count (an under-determined system cannot be
    # solved) and by the prefix length, which is the design delay spread.
    p = cfg.pilot_idx.size
    L = min(n_taps if n_taps else max(cfg.n_cp, 1), p)
    A = np.exp(-2j * np.pi * np.outer(cfg.pilot_idx, np.arange(L)) / cfg.n_sc)
    taps = np.linalg.lstsq(A, h_p.T, rcond=None)[0].T        # (n_blocks, L)
    return np.fft.fft(taps, cfg.n_sc, axis=1)


def demodulate(samples: np.ndarray, cfg: OFDMConfig,
               channel_taps: np.ndarray | None = None,
               method: str = "dft") -> tuple[np.ndarray, np.ndarray]:
    """Time-domain samples -> bits. Returns (bits, equalised data symbols).

    `channel_taps` supplies perfect channel state information; omit it to use
    the pilot-based estimate instead. Having both available is what lets the
    estimation loss be measured separately from everything else.
    """
    a = qam(cfg.M)
    s = np.asarray(samples, dtype=complex)
    spb = cfg.samples_per_block
    n_blocks = s.size // spb
    if n_blocks == 0:
        raise ValueError("fewer samples than one OFDM block")
    blocks = s[: n_blocks * spb].reshape(n_blocks, spb)
    no_cp = blocks[:, cfg.n_cp:] if cfg.n_cp else blocks

    grid = np.fft.fft(no_cp, axis=1) / math.sqrt(cfg.n_sc)

    if channel_taps is not None:
        H = np.fft.fft(np.asarray(channel_taps, dtype=complex), cfg.n_sc)
        H = np.broadcast_to(H, (n_blocks, cfg.n_sc))
    else:
        H = estimate_channel(grid, cfg, method)

    # Zero-forcing. The guard against division by a deep null is not cosmetic:
    # an unprotected divide turns one faded subcarrier into inf and then NaN,
    # which propagates through the slicer and silently poisons the whole block.
    safe = np.where(np.abs(H) > 1e-12, H, 1e-12)
    eq = grid / safe

    data = eq[:, cfg.data_idx]
    return a.demodulate(data.ravel()), data


def per_subcarrier_ber(cfg: OFDMConfig, channel_taps: np.ndarray,
                       ebn0_db: float) -> float:
    """Exact predicted BER with a sufficient prefix and perfect CSI.

    Each data subcarrier is an independent flat AWGN channel with its own SNR
    |H_k|^2/sigma^2, so the block BER is the average of the per-subcarrier AWGN
    error rates. This is the prediction that makes the OFDM test sharp: it says
    what the frequency-selective BER should be, not merely that it should be
    worse than flat.
    """
    m = qam(cfg.M).bits_per_symbol
    H = np.fft.fft(np.asarray(channel_taps, dtype=complex), cfg.n_sc)
    sigma2 = noise_sigma2(cfg, ebn0_db)
    snr = np.abs(H[cfg.data_idx]) ** 2 / sigma2
    # theory takes Eb/N0; per-subcarrier Es/N0 = snr, so Eb/N0 = snr / m.
    ebn0_k = 10.0 * np.log10(np.maximum(snr / m, 1e-30))
    return float(np.mean(awgn_ber_square_qam(cfg.M, ebn0_k)))


class OFDMResult:
    __slots__ = ("ber", "n_bits", "n_errors", "symbols", "predicted")

    def __init__(self, ber: float, n_bits: int, n_errors: int,
                 symbols: np.ndarray, predicted: float | None) -> None:
        self.ber, self.n_bits, self.n_errors = ber, n_bits, n_errors
        self.symbols, self.predicted = symbols, predicted

    def __repr__(self) -> str:  # pragma: no cover
        return (f"OFDMResult(ber={self.ber:.4e}, n_bits={self.n_bits}, "
                f"predicted={self.predicted})")


def simulate(n_blocks: int, cfg: OFDMConfig, ebn0_db: float,
             channel_taps: np.ndarray | None = None, perfect_csi: bool = True,
             rng: np.random.Generator | int | None = None,
             method: str = "dft") -> OFDMResult:
    """One pass: generate, transmit through the channel, add noise, recover."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    taps = np.array([1.0 + 0.0j]) if channel_taps is None else np.asarray(channel_taps, complex)
    bits = rng.integers(0, 2, size=n_blocks * cfg.bits_per_block, dtype=np.uint8)
    tx = modulate(bits, cfg)

    # Convolve and keep the first N samples of each block period. The tail of
    # one block genuinely lands on the next, which is what the prefix absorbs.
    rx = np.convolve(tx, taps)[: tx.size]

    sigma = math.sqrt(noise_sigma2(cfg, ebn0_db) / 2.0)
    rx = rx + (rng.standard_normal(rx.size) + 1j * rng.standard_normal(rx.size)) * sigma

    rx_bits, syms = demodulate(rx, cfg, taps if perfect_csi else None, method)
    n = min(bits.size, rx_bits.size)
    errs = int(np.count_nonzero(bits[:n] != rx_bits[:n]))
    predicted = (per_subcarrier_ber(cfg, taps, ebn0_db)
                 if perfect_csi and cfg.n_cp >= taps.size - 1 else None)
    return OFDMResult(errs / n, n, errs, syms, predicted)
