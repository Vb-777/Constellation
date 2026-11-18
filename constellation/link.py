"""The single-carrier link, assembled end to end.

transmit -> channel -> receive, with every stage optional so an impairment can
be isolated. The default path is the full one: Gray QAM, RRC shaping, a
fractional sampling offset and a carrier offset in the channel, then matched
filtering, Gardner timing recovery and a preamble-aided carrier loop.

The design rule throughout is that turning every impairment off must reproduce
the analytic BER to within Monte-Carlo error. Anything else means a dB has
leaked somewhere, and a link simulator that is 1 dB optimistic is worse than
none at all because it looks right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import channel, pulse, sync
from .modulation import ber as bit_error_rate
from .modulation import evm_db, qam, random_bits

__all__ = ["LinkConfig", "Impairments", "transmit", "receive", "simulate",
           "simulate_memoryless", "LinkResult", "ber_curve"]


@dataclass(frozen=True)
class LinkConfig:
    M: int = 4
    sps: int = 8
    beta: float = 0.22
    span: int = 10
    preamble_len: int = sync.PREAMBLE_LEN
    recover_timing: bool = True
    recover_carrier: bool = True
    timing_loop_bw: float = 0.005
    carrier_loop_bw: float = 0.01

    @property
    def alphabet(self):
        return qam(self.M)

    @property
    def bits_per_symbol(self) -> int:
        return self.alphabet.bits_per_symbol

    @property
    def taps(self) -> np.ndarray:
        return pulse.rrc(self.beta, self.span, self.sps)


@dataclass(frozen=True)
class Impairments:
    """What the channel does. All default to off so each can be isolated."""
    ebn0_db: float = np.inf
    timing_offset: float = 0.0        # samples, fractional
    freq_offset: float = 0.0          # cycles per symbol
    phase_offset: float = 0.0         # radians
    multipath_taps: np.ndarray | None = None
    rayleigh: bool = False
    #: Fade coherence length in *symbols*. It must be at least the pulse-shaping
    #: span for the flat-fading model to hold: the matched filter integrates
    #: over that many symbols, so a fade redrawn faster than that is
    #: time-selective and perfect-CSI correction by a single gain is not
    #: meaningful. The default is a quasi-static block, which is both realistic
    #: and the case the closed-form average BER describes.
    rayleigh_block: int = 50


@dataclass
class LinkResult:
    tx_bits: np.ndarray
    rx_bits: np.ndarray
    tx_symbols: np.ndarray
    rx_symbols: np.ndarray
    timing: sync.TimingResult | None = None
    carrier: sync.CarrierResult | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ber(self) -> float:
        return bit_error_rate(self.tx_bits, self.rx_bits)

    @property
    def evm_db(self) -> float:
        return evm_db(self.rx_symbols, self.tx_symbols)

    @property
    def n_bits(self) -> int:
        return min(self.tx_bits.size, self.rx_bits.size)

    @property
    def n_errors(self) -> int:
        n = self.n_bits
        return int(np.count_nonzero(self.tx_bits[:n] != self.rx_bits[:n]))


def transmit(bits: np.ndarray, cfg: LinkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map, prepend the preamble, upsample and pulse-shape.

    Returns (waveform, all_symbols, preamble).
    """
    a = cfg.alphabet
    payload = a.modulate(bits)
    pre = sync.bpsk_preamble(cfg.preamble_len) if cfg.preamble_len else np.empty(0, complex)
    syms = np.concatenate([pre, payload])
    wave = pulse.apply_filter(pulse.upsample(syms, cfg.sps), cfg.taps, mode="full")
    return wave, syms, pre


def _impair(wave: np.ndarray, cfg: LinkConfig, imp: Impairments,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    gains = None
    if imp.multipath_taps is not None:
        wave = channel.multipath(wave, imp.multipath_taps)
    if imp.rayleigh:
        wave, gains = channel.rayleigh_flat(wave, rng, block=imp.rayleigh_block * cfg.sps)
    if imp.timing_offset:
        wave = channel.fractional_delay(wave, imp.timing_offset)
    if imp.freq_offset or imp.phase_offset:
        # freq_offset is quoted per symbol; the waveform is per sample.
        wave = channel.frequency_offset(wave, imp.freq_offset / cfg.sps, imp.phase_offset)
    if np.isfinite(imp.ebn0_db):
        wave = channel.awgn(wave, imp.ebn0_db, cfg.bits_per_symbol, rng)
    return wave, gains


def receive(wave: np.ndarray, cfg: LinkConfig, n_symbols: int,
            preamble: np.ndarray, gains: np.ndarray | None = None) -> LinkResult:
    """Matched filter, strobe, derotate, slice."""
    a = cfg.alphabet
    taps = cfg.taps
    mf = pulse.apply_filter(wave, taps, mode="full")

    # Each length-L filter delays the pulse peak by (L-1)/2, so symbol k peaks
    # at (L-1) + k*sps in the matched-filter output. The Gardner loop takes its
    # first strobe one symbol in - it needs a previous sample for the error
    # signal - so the slice keeps one symbol of lead-in, which makes strobe k
    # nominally symbol k.
    delay = max(taps.size - 1 - cfg.sps, 0)
    aligned = mf[delay:]

    timing = None
    if cfg.recover_timing:
        timing = sync.timing_recovery(aligned, cfg.sps, cfg.timing_loop_bw)
        strobes = timing.symbols
    else:
        strobes = aligned[cfg.sps::cfg.sps][:n_symbols]

    # Nominal is not good enough. Correlate for the real frame start, so a
    # change to the filter span or the oversampling factor cannot silently slip
    # the preamble and send the carrier estimate chasing noise.
    frame_offset, frame_ratio = 0, np.inf
    if preamble.size:
        frame_offset, frame_ratio = sync.find_preamble(strobes, preamble)
        strobes = strobes[frame_offset:]

    if gains is not None:
        # Perfect CSI: undo the fade. Measuring a fading BER without this
        # measures a receiver with no channel estimate at all, which is a
        # different and much worse curve.
        #
        # The gain has to be sampled where the symbol's pulse actually sits.
        # The fade multiplies the transmit waveform, and symbol k's pulse is
        # centred at (L-1)/2 + k*sps there - not at k*sps, which is what a
        # plain `gains[::sps]` would take. That off-by-a-filter-delay indexing
        # divides each symbol by some other symbol's fade and pins the BER at
        # 0.5 while every loop in the receiver still reports a clean lock.
        #
        # This is only the right correction at all if the fade is constant
        # across the pulse. The matched filter spans `span` symbols, so a fade
        # redrawn every symbol is time-selective and no single gain describes
        # it; `Impairments.rayleigh_block` must cover the filter span for the
        # flat-fading model - and the closed form in `theory` - to apply.
        centre = (taps.size - 1) // 2
        idx = np.minimum(centre + np.arange(strobes.size) * cfg.sps, gains.size - 1)
        g = gains[idx]
        strobes = strobes / np.where(np.abs(g) > 1e-12, g, 1.0)

    carrier = None
    if cfg.recover_carrier:
        carrier = sync.carrier_recovery(strobes, a, preamble, cfg.carrier_loop_bw)
        strobes = carrier.symbols

    payload = strobes[preamble.size:][: n_symbols - preamble.size]
    return LinkResult(
        tx_bits=np.empty(0, np.uint8), rx_bits=a.demodulate(payload),
        tx_symbols=np.empty(0, complex), rx_symbols=payload,
        timing=timing, carrier=carrier,
        meta={"frame_offset": frame_offset, "frame_ratio": frame_ratio},
    )


def simulate(n_bits: int, cfg: LinkConfig = LinkConfig(),
             imp: Impairments = Impairments(),
             rng: np.random.Generator | int | None = None) -> LinkResult:
    """One burst through the whole chain."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    bps = cfg.bits_per_symbol
    n_bits = (n_bits // bps) * bps
    if n_bits <= 0:
        raise ValueError("need at least one symbol of payload")

    bits = random_bits(n_bits, rng)
    wave, syms, pre = transmit(bits, cfg)
    wave, gains = _impair(wave, cfg, imp, rng)
    res = receive(wave, cfg, syms.size, pre, gains)

    res.tx_bits = bits
    res.tx_symbols = syms[pre.size:][: res.rx_symbols.size]
    res.meta.update({"n_symbols": syms.size, "ebn0_db": imp.ebn0_db})
    return res


def simulate_memoryless(n_bits: int, M: int = 4, ebn0_db: float = 10.0,
                        rayleigh: bool = False,
                        rng: np.random.Generator | int | None = None) -> LinkResult:
    """Symbols straight through the channel: no pulse shaping, no loops.

    This is the model the closed forms in `theory` describe - one symbol, one
    channel gain, no memory - and it is the right place to validate them. It is
    not a shortcut around the full link; it answers a different question.

    It matters most for fading. A flat Rayleigh gain redrawn every symbol is
    *not* a flat channel once the signal is pulse-shaped: the matched filter
    integrates over the filter span, so neighbouring symbols with different
    gains leak into each other and the channel has memory the closed form does
    not model. Measured on the waveform link, a 40-symbol fade block sits 2.2x
    above the analytic curve at 18 dB, purely from symbols near a fade
    boundary. Here the agreement is 0.6% from 6 dB to 22 dB.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    a = qam(M)
    n_bits = (n_bits // a.bits_per_symbol) * a.bits_per_symbol
    if n_bits <= 0:
        raise ValueError("need at least one symbol")

    bits = random_bits(n_bits, rng)
    tx = a.modulate(bits)
    y = tx
    if rayleigh:
        y, h = channel.rayleigh_flat(y, rng, block=1)
        y = channel.awgn(y, ebn0_db, a.bits_per_symbol, rng) / h
    else:
        y = channel.awgn(y, ebn0_db, a.bits_per_symbol, rng)

    return LinkResult(tx_bits=bits, rx_bits=a.demodulate(y),
                      tx_symbols=tx, rx_symbols=y,
                      meta={"ebn0_db": ebn0_db, "memoryless": True})


def ber_curve(ebn0_db: np.ndarray, cfg: LinkConfig = LinkConfig(),
              imp: Impairments = Impairments(),
              target_errors: int = 200, min_bits: int = 20_000,
              max_bits: int = 4_000_000,
              rng: np.random.Generator | int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sweep Eb/N0, accumulating bursts until enough errors for a stable estimate.

    Stopping on an error count rather than a bit count is what keeps the high-SNR
    end of the curve meaningful: a fixed budget of bits gives a confidence
    interval that widens exactly where the curve gets interesting, and produces
    the ragged tails that make simulated BER plots untrustworthy.

    Returns (ber, bits_used). A point that exhausts `max_bits` without reaching
    `target_errors` is reported as measured, including zero, and the companion
    bit count is what lets the caller see it is an upper bound rather than a
    measurement.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    ebn0_db = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    out = np.empty(ebn0_db.size)
    used = np.empty(ebn0_db.size, dtype=np.int64)

    for i, db in enumerate(ebn0_db):
        errors = 0
        bits = 0
        burst = max(min_bits, 2000)
        while bits < max_bits and (errors < target_errors or bits < min_bits):
            r = simulate(burst, cfg, Impairments(**{**imp.__dict__, "ebn0_db": float(db)}), rng)
            errors += r.n_errors
            bits += r.n_bits
        out[i] = errors / bits if bits else np.nan
        used[i] = bits

    return out, used
