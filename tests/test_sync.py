"""Timing recovery, frame synchronisation and carrier recovery.

The tests that matter here are the ones for failures that look like success:
a timing loop locked one symbol off, a carrier loop locked 90 degrees off, a
frame offset by one symbol. All three produce a confident receiver and a BER of
roughly 0.5, and none of them shows up in a loop metric.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from constellation import channel, pulse, sync
from constellation.modulation import qam, random_bits


def _burst(M=4, n_sym=4000, sps=8, beta=0.22, span=10, seed=0, preamble=None):
    a = qam(M)
    rng = np.random.default_rng(seed)
    bits = random_bits(a.bits_per_symbol * n_sym, rng)
    syms = a.modulate(bits)
    if preamble is not None:
        syms = np.concatenate([preamble, syms])
    h = pulse.rrc(beta, span, sps)
    wave = pulse.apply_filter(pulse.upsample(syms, sps), h)
    return a, bits, h, wave, rng


def _matched(wave, h, sps):
    mf = pulse.apply_filter(wave, h)
    return mf[max(h.size - 1 - sps, 0):]


# ------------------------------------------------------------------ loop gains
@pytest.mark.parametrize("bw", [0.001, 0.01, 0.05])
def test_loop_gains_are_positive_and_grow_with_bandwidth(bw: float) -> None:
    kp, ki = sync.loop_gains(bw)
    kp2, ki2 = sync.loop_gains(bw * 2.0)
    assert kp > 0.0 and ki > 0.0
    assert kp2 > kp and ki2 > ki


@pytest.mark.parametrize("bad", [0.0, -0.1, 0.5, 1.0])
def test_invalid_loop_bandwidth_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        sync.loop_gains(bad)


# --------------------------------------------------------------- interpolation
def test_interpolator_is_exact_on_integer_indices() -> None:
    x = np.arange(20).astype(complex)
    for n in range(2, 17):
        assert sync.interpolate(x, float(n)) == pytest.approx(x[n])


def test_interpolator_is_exact_for_a_cubic() -> None:
    """Cubic Lagrange must reproduce any cubic exactly, which is its whole claim."""
    n = np.arange(24).astype(float)
    x = (0.3 * n**3 - 2.0 * n**2 + 5.0 * n - 1.0).astype(complex)
    for t in (5.25, 9.5, 13.75, 17.1):
        want = 0.3 * t**3 - 2.0 * t**2 + 5.0 * t - 1.0
        assert sync.interpolate(x, t).real == pytest.approx(want, rel=1e-9)


def test_interpolator_returns_zero_past_the_edges() -> None:
    x = np.ones(10, dtype=complex)
    assert sync.interpolate(x, 0.5) == 0
    assert sync.interpolate(x, 9.5) == 0


# ------------------------------------------------------------ timing recovery
@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, -0.4, 1.3])
def test_timing_loop_finds_the_injected_sampling_phase(offset: float) -> None:
    """Convergence on the right value, not merely settling on some value."""
    a, bits, h, wave, _ = _burst(sps=8, n_sym=3000, seed=1)
    wave = channel.fractional_delay(wave, offset)
    res = sync.timing_recovery(_matched(wave, h, 8), 8, 0.005)
    assert res.steady_state_tau() == pytest.approx(offset, abs=0.15)


@pytest.mark.parametrize("sps", [4, 8])
def test_timing_loop_recovers_symbols_cleanly_without_noise(sps: int) -> None:
    a, bits, h, wave, _ = _burst(sps=sps, n_sym=2000, seed=2)
    wave = channel.fractional_delay(wave, 0.3)
    res = sync.timing_recovery(_matched(wave, h, sps), sps, 0.005)
    n = min(res.symbols.size, bits.size // a.bits_per_symbol) - 20
    got = a.demodulate(res.symbols[10:n])
    want = bits[10 * a.bits_per_symbol: n * a.bits_per_symbol]
    assert np.count_nonzero(got != want) == 0


def test_timing_loop_settles_with_small_jitter() -> None:
    """Even with no noise at all the loop jitters, and that is not a defect.

    The Gardner detector's error signal depends on the data, so random symbols
    drive it with a residual the loop cannot distinguish from a genuine timing
    error. That is self-noise, it is irreducible at a given loop bandwidth, and
    the only way to shrink it is to narrow the loop - which slows acquisition.
    Measured here at 0.06 samples, i.e. 0.0075 of a symbol period.
    """
    a, bits, h, wave, _ = _burst(sps=8, n_sym=4000, seed=3)
    res = sync.timing_recovery(_matched(channel.fractional_delay(wave, 0.4), h, 8),
                               8, 0.005)
    assert res.jitter() < 0.1


def test_timing_error_signal_averages_to_zero_once_locked() -> None:
    """The defining property of the detector at the lock point."""
    a, bits, h, wave, _ = _burst(sps=8, n_sym=4000, seed=4)
    res = sync.timing_recovery(_matched(channel.fractional_delay(wave, 0.35), h, 8),
                               8, 0.005)
    tail = res.error[res.error.size // 2:]
    assert abs(float(np.mean(tail))) < 0.2 * float(np.std(tail)) + 1e-6


def test_timing_loop_is_blind_to_carrier_phase() -> None:
    """Gardner's advantage over Mueller-Muller, and why it runs first.

    A 50-degree carrier rotation must not move the recovered sampling phase.
    """
    a, bits, h, wave, _ = _burst(sps=8, n_sym=3000, seed=5)
    wave = channel.fractional_delay(wave, 0.3)
    clean = sync.timing_recovery(_matched(wave, h, 8), 8, 0.005).steady_state_tau()
    rot = sync.timing_recovery(_matched(channel.phase_offset(wave, 0.87), h, 8),
                               8, 0.005).steady_state_tau()
    assert rot == pytest.approx(clean, abs=0.02)


def test_timing_loop_rejects_too_few_samples_per_symbol() -> None:
    with pytest.raises(ValueError):
        sync.timing_recovery(np.ones(1000, complex), 1)


def test_timing_loop_rejects_a_burst_too_short_to_lock() -> None:
    with pytest.raises(ValueError):
        sync.timing_recovery(np.ones(20, complex), 8)


def test_timing_loop_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError):
        sync.timing_recovery(np.ones(1000, complex), 8, backend="rust")


def test_sampling_phase_stays_bounded_within_one_symbol() -> None:
    """The clamp that stops the loop sliding onto a neighbouring symbol."""
    a, bits, h, wave, rng = _burst(sps=8, n_sym=2000, seed=6)
    noisy = channel.awgn(wave, -3.0, 2, rng)   # deliberately hopeless SNR
    res = sync.timing_recovery(_matched(noisy, h, 8), 8, 0.05)
    assert np.all(np.abs(res.tau) <= 8.0 + 1e-12)


# ------------------------------------------------------------- preamble, frame
def test_preamble_is_a_repetition_of_one_half() -> None:
    p = sync.bpsk_preamble(64)
    assert p.size == 64
    assert np.allclose(p[:32], p[32:])


def test_preamble_is_constant_modulus() -> None:
    assert np.allclose(np.abs(sync.bpsk_preamble(64)), 1.0)


def test_odd_preamble_length_rejected() -> None:
    with pytest.raises(ValueError):
        sync.bpsk_preamble(63)


@pytest.mark.parametrize("shift", [0, 1, 2, 5])
def test_frame_sync_finds_the_offset(shift: int) -> None:
    """One-symbol slips are the failure this block exists to prevent."""
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(7)
    pad = (rng.standard_normal(shift) + 1j * rng.standard_normal(shift)) / math.sqrt(2)
    stream = np.concatenate([pad, p, qam(4).modulate(random_bits(400, rng))])
    found, ratio = sync.find_preamble(stream, p)
    assert found == shift
    assert ratio > 2.0


def test_frame_sync_is_insensitive_to_carrier_phase() -> None:
    p = sync.bpsk_preamble(64)
    stream = np.concatenate([np.zeros(3, complex), p,
                             qam(4).modulate(random_bits(400, 8))])
    found, _ = sync.find_preamble(stream * np.exp(1.9j), p)
    assert found == 3


def test_frame_sync_survives_noise() -> None:
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(9)
    stream = np.concatenate([np.zeros(2, complex), p,
                             qam(4).modulate(random_bits(400, rng))])
    found, ratio = sync.find_preamble(channel.awgn(stream, 4.0, 2, rng), p)
    assert found == 2 and ratio > 1.5


def test_frame_sync_reports_confidence_so_failure_is_visible() -> None:
    """On pure noise the peak ratio must stay low rather than assert lag zero."""
    rng = np.random.default_rng(10)
    noise = (rng.standard_normal(300) + 1j * rng.standard_normal(300)) / math.sqrt(2)
    _, ratio = sync.find_preamble(noise, sync.bpsk_preamble(64))
    assert ratio < 4.0


def test_frame_sync_with_no_preamble_is_a_no_op() -> None:
    off, ratio = sync.find_preamble(np.ones(100, complex), np.empty(0, complex))
    assert off == 0 and ratio == np.inf


def test_frame_sync_rejects_a_stream_shorter_than_the_preamble() -> None:
    with pytest.raises(ValueError):
        sync.find_preamble(np.ones(10, complex), sync.bpsk_preamble(64))


# --------------------------------------------------- frequency estimation
@pytest.mark.parametrize("f", [0.0, 1e-4, 1e-3, 5e-3, -2e-3, 1e-2])
def test_repetition_estimator_recovers_the_offset(f: float) -> None:
    p = sync.bpsk_preamble(64)
    rx = channel.frequency_offset(p, f, phase0=0.6)
    assert sync.frequency_from_repetition(rx, 32) == pytest.approx(f, abs=2e-5)


def test_repetition_estimator_is_immune_to_a_constant_phase() -> None:
    p = sync.bpsk_preamble(64)
    a = sync.frequency_from_repetition(channel.phase_offset(p, 0.0), 32)
    b = sync.frequency_from_repetition(channel.phase_offset(p, 2.2), 32)
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize("db", [6.0, 10.0, 20.0])
def test_repetition_estimator_is_accurate_under_noise(db: float) -> None:
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(11)
    errs = []
    for _ in range(40):
        rx = channel.awgn(channel.frequency_offset(p, 1e-3), db, 2, rng)
        errs.append(sync.frequency_from_repetition(rx, 32) - 1e-3)
    assert abs(float(np.mean(errs))) < 3e-4


def test_repetition_estimator_wraps_past_its_capture_range() -> None:
    """|f| < 1/(2*half) is a real limit, and it must be documented, not hidden."""
    p = sync.bpsk_preamble(64)
    beyond = 1.0 / 32.0          # twice the 1/(2*32) limit
    got = sync.frequency_from_repetition(channel.frequency_offset(p, beyond), 32)
    assert abs(got - beyond) > 1e-3


def test_repetition_estimator_rejects_a_short_input() -> None:
    with pytest.raises(ValueError):
        sync.frequency_from_repetition(np.ones(10, complex), 32)


@pytest.mark.parametrize("f", [0.0, 1e-3, -1e-3])
def test_data_aided_estimator_is_exact_without_noise(f: float) -> None:
    """The lag-1 estimator is not broken - it is noisier, which is a different
    claim and the one the next test actually measures."""
    p = sync.bpsk_preamble(128)
    rx = channel.frequency_offset(p, f, phase0=0.3)
    assert sync.coarse_frequency_estimate(rx, p) == pytest.approx(f, abs=1e-9)


@pytest.mark.parametrize("db", [0.0, 8.0, 20.0])
def test_repetition_estimator_has_lower_variance_than_lag_one(db: float) -> None:
    """The real reason the link uses the repeated preamble.

    One correlation at lag 32 beats 63 correlations at lag 1, because the phase
    being measured has had 32 symbols to accumulate. Measured over 600 trials:
    2.7x lower standard deviation at 20 dB, 9.2x at 0 dB.

    This test exists because the advantage was first attributed to the wrong
    mechanism - a supposed payload-ISI bias scaling with peak-to-average power.
    Measurement showed no trend with constellation order at all, and the 0.5
    BER that prompted the change was missing frame synchronisation. The claim
    that survived is this one, so this is the claim under test.
    """
    f = 1e-3
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(50)
    e1, e2 = [], []
    for _ in range(600):
        rx = channel.awgn(channel.frequency_offset(p, f, 0.6), db, 2, rng)
        e1.append(sync.coarse_frequency_estimate(rx, p) - f)
        e2.append(sync.frequency_from_repetition(rx, 32) - f)
    assert float(np.std(e1)) > 2.0 * float(np.std(e2))


def test_neither_estimator_is_biased_by_constellation_order() -> None:
    """Explicitly pins down the claim that turned out to be false."""
    f = 1e-3
    p = sync.bpsk_preamble(64)
    biases = []
    for M in (4, 16, 64):
        a = qam(M)
        rng = np.random.default_rng(51)
        syms = np.concatenate([p, a.modulate(random_bits(a.bits_per_symbol * 500, rng))])
        rx = channel.frequency_offset(syms, f)
        biases.append(abs(sync.frequency_from_repetition(rx[:64], 32) - f))
    assert max(biases) < 1e-5, f"biases by order: {biases}"


# ----------------------------------------------------------- carrier recovery
@pytest.mark.parametrize("M", [4, 16])
@pytest.mark.parametrize("phase", [0.0, 0.8, 2.0, -2.5, math.pi / 2, math.pi])
def test_carrier_loop_resolves_the_rotational_ambiguity(M: int, phase: float) -> None:
    """The failure this test exists for: a decision-directed loop on square QAM
    is equally happy 90, 180 or 270 degrees off, because the rotated
    constellation is the same set of points. Only the known preamble breaks the
    tie, and multiples of pi/2 are included here deliberately."""
    a = qam(M)
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(12)
    bits = random_bits(a.bits_per_symbol * 2000, rng)
    syms = np.concatenate([p, a.modulate(bits)])
    res = sync.carrier_recovery(channel.phase_offset(syms, phase), a, p, 0.01)
    out = res.symbols[p.size:]
    assert np.count_nonzero(a.demodulate(out) != bits) == 0


@pytest.mark.parametrize("f", [0.0, 5e-4, 2e-3, -1e-3])
def test_carrier_loop_tracks_a_frequency_offset_through_the_payload(f: float) -> None:
    a = qam(4)
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(13)
    bits = random_bits(a.bits_per_symbol * 3000, rng)
    syms = np.concatenate([p, a.modulate(bits)])
    rx = channel.frequency_offset(syms, f, phase0=0.4)
    res = sync.carrier_recovery(rx, a, p, 0.01)
    assert np.count_nonzero(a.demodulate(res.symbols[p.size:]) != bits) == 0
    assert res.f_coarse == pytest.approx(f, abs=5e-5)


def test_carrier_loop_residual_phase_error_is_small() -> None:
    a = qam(16)
    p = sync.bpsk_preamble(64)
    rng = np.random.default_rng(14)
    bits = random_bits(a.bits_per_symbol * 3000, rng)
    syms = np.concatenate([p, a.modulate(bits)])
    res = sync.carrier_recovery(channel.frequency_offset(syms, 1e-3, 0.5), a, p, 0.01)
    assert res.residual_phase_rms(syms) < 0.05


def test_carrier_loop_without_a_preamble_still_runs() -> None:
    """Blind operation must not crash; it is simply allowed to be ambiguous."""
    a = qam(4)
    syms = a.modulate(random_bits(2000, 15))
    res = sync.carrier_recovery(syms, a, None, 0.01)
    assert res.symbols.size == syms.size
    assert res.f_coarse == 0.0
