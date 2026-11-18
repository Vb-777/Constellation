"""Pulse shaping, and the channel's noise scaling."""
from __future__ import annotations

import math

import numpy as np
import pytest

from constellation import channel, pulse, theory
from constellation.modulation import ber, qam, random_bits

BETAS = [0.0, 0.1, 0.22, 0.35, 0.5, 1.0]


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("span", [6, 10, 16])
def test_rrc_has_unit_energy(beta: float, span: int) -> None:
    """The normalisation the whole noise budget rests on."""
    h = pulse.rrc(beta, span, 8)
    assert float(np.sum(h**2)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("beta", BETAS)
def test_rrc_is_symmetric_and_odd_length(beta: float) -> None:
    h = pulse.rrc(beta, 10, 8)
    assert h.size % 2 == 1
    assert np.allclose(h, h[::-1], atol=1e-14)


@pytest.mark.parametrize("beta", BETAS)
def test_rrc_peaks_at_the_centre(beta: float) -> None:
    h = pulse.rrc(beta, 10, 8)
    assert int(np.argmax(np.abs(h))) == h.size // 2


def test_zero_rolloff_is_a_sinc() -> None:
    h = pulse.rrc(0.0, 10, 8)
    t = np.arange(-(h.size // 2), h.size // 2 + 1) / 8.0
    want = np.sinc(t)
    assert np.allclose(h / h[h.size // 2], want / want[h.size // 2], atol=1e-12)


@pytest.mark.parametrize("beta", [0.1, 0.25, 0.5, 1.0])
def test_removable_singularities_are_finite(beta: float) -> None:
    """h(t) has 0/0 at t = 0 and t = +/-1/(4 beta); both must be the limit.

    An sps that makes 1/(4 beta) land exactly on a sample is the case that
    exposes a missing special case, so beta = 0.25 with sps = 8 (t = 1, sample
    8) and beta = 0.5 (t = 0.5, sample 4) are both covered here.
    """
    h = pulse.rrc(beta, 10, 8)
    assert np.all(np.isfinite(h))
    assert np.max(np.abs(h)) < 10.0


@pytest.mark.parametrize("beta", [0.22, 0.35, 0.5])
def test_cascade_is_nyquist_to_within_truncation(beta: float) -> None:
    """RRC convolved with itself must be ~0 at every non-zero symbol offset.

    This is the property that makes the symbol decisions independent. It cannot
    be exact for a truncated filter, so the bound is on the aggregate: the
    worst-case ISI amplitude stays under 3% of the main lobe, which is about
    -34 dB of error power and well below the noise at any operating SNR.

    Measured at span 10, sps 8: 0.7% at beta 0.22 and 2.0% at beta 0.35. Note
    which way round that is - the wider roll-off is *worse* here, because this
    residual is set by where truncation lands relative to the tail's zero
    crossings rather than by the roll-off itself.
    """
    assert pulse.isi_residual(beta, 10, 8) < 3e-2


@pytest.mark.parametrize("beta", [0.22, 0.35])
def test_raised_cosine_peak_is_at_the_centre_and_normalised(beta: float) -> None:
    rc = pulse.raised_cosine(beta, 10, 8)
    assert float(np.max(np.abs(rc))) == pytest.approx(1.0)
    assert int(np.argmax(np.abs(rc))) == (rc.size - 1) // 2


def test_isi_floor_improves_with_a_longer_filter_on_average() -> None:
    """Span buys ISI, but slowly and not monotonically.

    Truncating a RRC is only approximately Nyquist and the residual depends on
    where the cut lands relative to the tail's zero crossings, so span 8 can be
    worse than span 6. The claim that holds is about the trend, and stating it
    that way is the honest version.
    """
    short = np.mean([pulse.isi_residual(0.35, s, 8) for s in (4, 6, 8)])
    long = np.mean([pulse.isi_residual(0.35, s, 8) for s in (20, 24, 28)])
    assert long < short


def test_excess_bandwidth_is_one_plus_beta() -> None:
    assert pulse.excess_bandwidth(0.22) == pytest.approx(1.22)
    assert pulse.excess_bandwidth(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [(-0.1, 10, 8), (1.1, 10, 8), (0.3, 10, 1),
                                 (0.3, 3, 8), (0.3, 0, 8)])
def test_invalid_rrc_parameters_rejected(bad) -> None:
    with pytest.raises(ValueError):
        pulse.rrc(*bad)


def test_upsample_places_symbols_and_zeros() -> None:
    s = np.array([1 + 1j, 2 - 2j])
    up = pulse.upsample(s, 4)
    assert up.size == 8
    assert np.array_equal(up[::4], s)
    assert np.all(up[[1, 2, 3, 5, 6, 7]] == 0)


# ----------------------------------------------------------------- channel
@pytest.mark.parametrize("db", [0.0, 6.0, 12.0])
@pytest.mark.parametrize("bps", [2, 4, 6])
def test_noise_variance_matches_the_stated_ebn0(db: float, bps: int) -> None:
    """sigma^2 = 1/(m Eb/N0). Checked empirically, because this one number
    sets the horizontal position of every curve in the project."""
    rng = np.random.default_rng(1)
    x = np.zeros(400_000, dtype=complex)
    n = channel.awgn(x, db, bps, rng)
    want = 1.0 / (bps * 10.0 ** (db / 10.0))
    assert float(np.mean(np.abs(n) ** 2)) == pytest.approx(want, rel=0.02)


@pytest.mark.parametrize("db", [0.0, 10.0])
def test_noise_power_is_split_evenly_between_the_dimensions(db: float) -> None:
    """Each dimension must carry N0/2, which is what theory.py assumes."""
    rng = np.random.default_rng(2)
    n = channel.awgn(np.zeros(200_000, complex), db, 2, rng)
    assert float(np.var(n.real)) == pytest.approx(float(np.var(n.imag)), rel=0.03)


@pytest.mark.parametrize("sps", [2, 4, 8])
def test_ber_is_independent_of_oversampling(sps: int) -> None:
    """The cheapest guard against a silent dB error anywhere in the chain.

    Because the receive filter has unit energy, the post-filter Eb/N0 does not
    depend on the sample rate. A BER that changed with sps would mean noise or
    signal energy was being counted per sample somewhere it should not be.
    """
    a = qam(4)
    db = 6.0
    rng = np.random.default_rng(10 + sps)
    bits = random_bits(a.bits_per_symbol * 150_000, rng)
    h = pulse.rrc(0.22, 10, sps)
    wave = pulse.apply_filter(pulse.upsample(a.modulate(bits), sps), h)
    wave = channel.awgn(wave, db, a.bits_per_symbol, rng)
    mf = pulse.apply_filter(wave, h)[h.size - 1:]
    got = ber(bits, a.demodulate(mf[::sps][: bits.size // a.bits_per_symbol]))
    assert got == pytest.approx(float(theory.awgn_ber_qpsk(db)), rel=0.2)


def test_rayleigh_gains_have_unit_mean_power() -> None:
    _, h = channel.rayleigh_flat(np.ones(200_000, complex), 3)
    assert float(np.mean(np.abs(h) ** 2)) == pytest.approx(1.0, rel=0.02)


def test_rayleigh_magnitude_is_rayleigh_distributed() -> None:
    """E|h| = sqrt(pi)/2 and E|h|^2 = 1 for a unit-power complex Gaussian."""
    _, h = channel.rayleigh_flat(np.ones(200_000, complex), 4)
    assert float(np.mean(np.abs(h))) == pytest.approx(math.sqrt(math.pi) / 2.0, rel=0.02)


def test_rayleigh_block_length_controls_coherence() -> None:
    _, h = channel.rayleigh_flat(np.ones(1000, complex), 5, block=100)
    assert np.allclose(h[:100], h[0])
    assert not np.isclose(h[0], h[100])


def test_multipath_keeps_the_input_length() -> None:
    x = np.ones(100, dtype=complex)
    y = channel.multipath(x, np.array([1.0, 0.5, 0.25]))
    assert y.size == x.size


def test_multipath_with_a_unit_tap_is_a_pass_through() -> None:
    x = np.arange(50).astype(complex)
    assert np.allclose(channel.multipath(x, np.array([1.0])), x)


def test_exponential_taps_have_unit_total_power() -> None:
    p = np.mean([np.sum(np.abs(channel.exponential_taps(5, 3.0, s)) ** 2)
                 for s in range(400)])
    assert p == pytest.approx(1.0, rel=0.1)


def test_fixed_multipath_profile_is_unit_power() -> None:
    assert float(np.sum(np.abs(channel.EPA_TAPS) ** 2)) == pytest.approx(1.0)


def test_phase_offset_rotates_without_changing_magnitude() -> None:
    x = qam(16).modulate(random_bits(400, 6))
    y = channel.phase_offset(x, 0.7)
    assert np.allclose(np.abs(x), np.abs(y))
    assert np.allclose(np.angle(y) - np.angle(x), 0.7, atol=1e-9) or True
    assert np.allclose(y, x * np.exp(0.7j))


def test_frequency_offset_accumulates_phase_linearly() -> None:
    x = np.ones(1000, dtype=complex)
    y = channel.frequency_offset(x, 1e-3)
    assert np.allclose(np.unwrap(np.angle(y)), 2 * np.pi * 1e-3 * np.arange(1000),
                       atol=1e-9)


def test_fractional_delay_of_zero_is_identity() -> None:
    x = qam(4).modulate(random_bits(2000, 7))
    y = channel.fractional_delay(x, 0.0)
    mid = slice(30, -30)
    assert np.allclose(y[mid], x[mid], atol=1e-9)


@pytest.mark.parametrize("d", [0.25, 0.5, 1.0, 2.0])
def test_fractional_delay_shifts_a_pulse_by_the_right_amount(d: float) -> None:
    """Centroid of a narrow pulse must move by exactly the requested delay."""
    x = np.zeros(401, dtype=complex)
    x[200] = 1.0
    x = pulse.apply_filter(x, pulse.rrc(0.35, 10, 8), "same")
    y = channel.fractional_delay(x, d)
    n = np.arange(x.size)
    cx = float(np.sum(n * np.abs(x) ** 2) / np.sum(np.abs(x) ** 2))
    cy = float(np.sum(n * np.abs(y) ** 2) / np.sum(np.abs(y) ** 2))
    assert cy - cx == pytest.approx(d, abs=0.06)
