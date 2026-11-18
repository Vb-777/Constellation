"""Root-raised-cosine pulse shaping, and the Nyquist property that justifies it.

A rectangular pulse is spectrally hopeless and a sinc is unrealisable, so real
links use a root-raised-cosine split evenly between transmitter and receiver.
Two things then happen at once: the receive filter is matched to the transmit
pulse, so it maximises SNR, and the *cascade* of the two is a full raised
cosine, which is zero at every non-zero multiple of the symbol period. That
second property is the one that matters here - it is what makes the symbol
decisions in `modulation.py` independent, and `test_pulse.py` measures the
residual rather than taking it on trust.

Taps are normalised to unit energy. That is a deliberate choice: with
sum(h^2) = 1 the matched filter passes the signal amplitude through unchanged
and multiplies the noise variance by one, so the post-filter Eb/N0 equals the
Eb/N0 that `channel.awgn` was asked for, independent of oversampling.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["rrc", "raised_cosine", "upsample", "apply_filter", "isi_profile",
           "isi_residual", "isi_floor_db", "excess_bandwidth"]


def rrc(beta: float, span: int, sps: int) -> np.ndarray:
    """Unit-energy root-raised-cosine taps.

    beta  roll-off in [0, 1]; 0 is a sinc, 1 is maximally smooth
    span  filter length in symbol periods (even, so the filter is centred)
    sps   samples per symbol
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"roll-off must be in [0, 1], got {beta}")
    if sps < 2:
        raise ValueError(f"need at least 2 samples per symbol, got {sps}")
    if span < 2 or span % 2:
        raise ValueError(f"span must be even and >= 2 symbols, got {span}")

    n = span * sps
    t = np.arange(-n // 2, n // 2 + 1, dtype=float) / sps   # time in symbol periods
    h = np.empty_like(t)

    if beta == 0.0:
        h = np.sinc(t)
    else:
        # The closed form has removable singularities at t = 0 and t = +/-1/(4b).
        quarter = 1.0 / (4.0 * beta)
        at_zero = np.isclose(t, 0.0, atol=1e-12)
        at_quarter = np.isclose(np.abs(t), quarter, atol=1e-12)
        ordinary = ~(at_zero | at_quarter)

        tt = t[ordinary]
        num = (np.sin(np.pi * tt * (1.0 - beta))
               + 4.0 * beta * tt * np.cos(np.pi * tt * (1.0 + beta)))
        den = np.pi * tt * (1.0 - (4.0 * beta * tt) ** 2)
        h[ordinary] = num / den

        h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
        h[at_quarter] = (beta / math.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * math.sin(np.pi / (4.0 * beta))
            - (1.0 - 2.0 / np.pi) * math.cos(np.pi / (4.0 * beta))
        )

    return h / math.sqrt(float(np.sum(h**2)))


def raised_cosine(beta: float, span: int, sps: int) -> np.ndarray:
    """The Tx-Rx cascade, obtained by convolving the RRC with itself.

    Returned peak-normalised, because the useful question about it is the size
    of the samples at non-zero symbol offsets relative to the main lobe.
    """
    h = rrc(beta, span, sps)
    rc = np.convolve(h, h)
    return rc / np.max(np.abs(rc))


def isi_profile(beta: float, span: int, sps: int) -> np.ndarray:
    """Peak-normalised |RC| at every non-zero multiple of the symbol period.

    All zero for an ideal infinite filter. For a truncated one these are the
    actual interfering contributions a neighbouring symbol makes to the
    decision instant.
    """
    rc = raised_cosine(beta, span, sps)
    centre = (rc.size - 1) // 2
    reach = centre // sps
    offsets = np.arange(-reach, reach + 1) * sps + centre
    offsets = offsets[offsets != centre]
    return np.abs(rc[offsets])


def isi_residual(beta: float, span: int, sps: int) -> float:
    """Peak distortion: the sum of |RC| over all non-zero symbol offsets.

    This is the worst-case ISI amplitude at the slicer, reached when every
    neighbouring symbol conspires to push the same way, and it is the standard
    figure of merit for a truncated Nyquist filter.

    The sum, not the maximum. Taking the maximum measures the wrong thing: for
    every span tried here the largest single term sits at the *outermost*
    symbol offset, where the ideal cascade is essentially zero and what remains
    is the discontinuity left by truncating the filter. That term is an edge
    artefact affecting the first and last symbols of a burst; the sum is what a
    symbol in the middle of a long transmission actually sees.

    The decay with span is slow and visibly non-monotonic, because a truncated
    RRC is only approximately Nyquist and the residual depends on where the cut
    lands relative to the zero crossings of the tail. Doubling the span does
    not reliably halve the ISI, which is the practical reason real modems
    settle at 8 to 12 symbols and spend the gates elsewhere.
    """
    p = isi_profile(beta, span, sps)
    return float(np.sum(p)) if p.size else 0.0


def isi_floor_db(beta: float, span: int, sps: int) -> float:
    """Residual ISI as an EVM-style power ratio in dB, which is the number that bites.

    Treats the interfering taps as independent contributions of random sign -
    appropriate for random data - so the power sums rather than the amplitudes.
    This is the error floor pulse shaping alone imposes, before any noise.
    """
    p = isi_profile(beta, span, sps)
    power = float(np.sum(p**2))
    return -np.inf if power == 0.0 else 10.0 * math.log10(power)


def excess_bandwidth(beta: float) -> float:
    """Occupied bandwidth as a multiple of the symbol rate: (1 + beta).

    The whole trade: beta = 0.22 (LTE's shaping is close to this) costs 22%
    more spectrum than Nyquist and buys a filter that can actually be built
    and a timing loop that can actually lock.
    """
    return 1.0 + beta


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Insert sps-1 zeros after each symbol."""
    s = np.asarray(symbols, dtype=complex)
    out = np.zeros(s.size * sps, dtype=complex)
    out[::sps] = s
    return out


def apply_filter(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=complex), np.asarray(h, dtype=float), mode)
