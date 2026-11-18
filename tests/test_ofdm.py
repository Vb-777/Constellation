"""OFDM: the cyclic prefix, the overhead accounting, and the two estimators.

The central test is `test_matches_the_per_subcarrier_prediction`. With a
sufficient prefix and perfect channel knowledge there is no residual ISI, so
the BER is forced to equal the average of the per-subcarrier AWGN error rates
at each subcarrier's own SNR. That is an exact prediction over a
frequency-selective channel, not an inequality, which makes cyclic-prefix
sufficiency the only thing left that can fail.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from constellation import channel, ofdm, theory

TAPS = channel.EPA_TAPS


def cfg(n_sc=64, n_cp=16, spacing=8, M=4):
    return ofdm.OFDMConfig(n_sc, n_cp, spacing, M)


# ----------------------------------------------------------------- layout
def test_pilots_and_data_partition_the_subcarriers() -> None:
    c = cfg()
    assert c.pilot_idx.size + c.n_data == c.n_sc
    assert not set(c.pilot_idx.tolist()) & set(c.data_idx.tolist())


def test_block_sizes_are_consistent() -> None:
    c = cfg()
    assert c.samples_per_block == c.n_sc + c.n_cp
    assert c.bits_per_block == c.n_data * 2


@pytest.mark.parametrize("bad", [(7, 2, 8, 4), (64, 64, 8, 4), (64, -1, 8, 4),
                                 (64, 16, 7, 4), (64, 16, 1, 4), (6, 2, 2, 4)])
def test_invalid_layouts_rejected(bad) -> None:
    with pytest.raises(ValueError):
        ofdm.OFDMConfig(*bad)


def test_overhead_is_the_log_of_the_wasted_fraction() -> None:
    """1.55 dB for the defaults. Forgetting this term is what makes an OFDM
    curve appear to beat single-carrier QAM, which cannot happen."""
    c = cfg()
    assert ofdm.overhead_db(c) == pytest.approx(
        10.0 * math.log10(80.0 / 56.0), rel=1e-12)
    assert ofdm.overhead_db(c) == pytest.approx(1.549, abs=0.001)


def test_overhead_grows_with_the_prefix_and_with_pilot_density() -> None:
    assert ofdm.overhead_db(cfg(n_cp=32)) > ofdm.overhead_db(cfg(n_cp=16))
    assert ofdm.overhead_db(cfg(spacing=4)) > ofdm.overhead_db(cfg(spacing=8))


# ------------------------------------------------------------- round trips
@pytest.mark.parametrize("M", [4, 16])
def test_noiseless_flat_round_trip_is_lossless(M: int) -> None:
    c = cfg(M=M)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=20 * c.bits_per_block, dtype=np.uint8)
    out, _ = ofdm.demodulate(ofdm.modulate(bits, c), c, np.array([1.0 + 0j]))
    assert np.array_equal(out, bits)


def test_noiseless_multipath_round_trip_is_lossless_with_a_long_prefix() -> None:
    """The point of the cyclic prefix, stated as an exact equality."""
    c = cfg()
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=20 * c.bits_per_block, dtype=np.uint8)
    tx = ofdm.modulate(bits, c)
    rx = np.convolve(tx, TAPS)[: tx.size]
    out, _ = ofdm.demodulate(rx, c, TAPS)
    assert np.array_equal(out, bits)


def test_cyclic_prefix_is_a_copy_of_the_block_tail() -> None:
    c = cfg()
    rng = np.random.default_rng(2)
    tx = ofdm.modulate(rng.integers(0, 2, size=c.bits_per_block, dtype=np.uint8), c)
    assert np.allclose(tx[: c.n_cp], tx[c.n_cp + c.n_sc - c.n_cp:])


def test_time_domain_samples_have_unit_mean_power() -> None:
    """The unitary IFFT scaling, which the noise budget depends on."""
    c = cfg()
    rng = np.random.default_rng(3)
    tx = ofdm.modulate(rng.integers(0, 2, size=400 * c.bits_per_block, dtype=np.uint8), c)
    assert float(np.mean(np.abs(tx) ** 2)) == pytest.approx(1.0, rel=0.03)


def test_bit_count_must_fill_whole_blocks() -> None:
    c = cfg()
    with pytest.raises(ValueError):
        ofdm.modulate(np.ones(c.bits_per_block + 1, dtype=np.uint8), c)


def test_too_few_samples_to_demodulate_is_an_error() -> None:
    with pytest.raises(ValueError):
        ofdm.demodulate(np.ones(5, complex), cfg())


# ------------------------------------------------- the central prediction
@pytest.mark.parametrize("db", [6.0, 10.0, 14.0, 18.0])
def test_matches_the_per_subcarrier_prediction(db: float) -> None:
    """Exact over a frequency-selective channel, within Monte-Carlo error."""
    c = cfg()
    r = ofdm.simulate(8000, c, db, TAPS, True, 20)
    assert r.predicted is not None
    assert r.ber == pytest.approx(r.predicted, rel=0.08)


@pytest.mark.parametrize("db", [4.0, 6.0, 8.0, 10.0])
def test_flat_channel_matches_awgn_theory_shifted_by_the_overhead(db: float) -> None:
    """Validates the energy bookkeeping to a fraction of a dB.

    Stops at 10 dB: past that the predicted BER falls below 1e-5 and the bit
    budget here would see single-digit error counts, so the test would start
    measuring the seed rather than the bookkeeping.
    """
    c = cfg()
    want = float(theory.awgn_ber_square_qam(c.M, db - ofdm.overhead_db(c)))
    n_blocks = min(int(400.0 / want / c.bits_per_block) + 2000, 60_000)
    r = ofdm.simulate(n_blocks, c, db, None, True, 21)
    assert r.ber == pytest.approx(want, rel=0.12)


def test_frequency_selectivity_costs_far_more_than_the_overhead() -> None:
    """Without coding across subcarriers, the deep fades dominate.

    OFDM converts one frequency-selective channel into many flat ones; it does
    not by itself provide any diversity. The subcarriers in nulls carry nearly
    all of the errors, so the curve bends towards the fading shape rather than
    the AWGN one. Coding across subcarriers is what recovers the diversity, and
    it is not implemented here.
    """
    c = cfg()
    flat = ofdm.simulate(6000, c, 18.0, None, True, 22).ber
    sel = ofdm.simulate(6000, c, 18.0, TAPS, True, 22).ber
    assert sel > 20.0 * max(flat, 1e-12)


# --------------------------------------------------------- prefix sufficiency
def test_prefix_at_least_the_delay_spread_removes_the_isi_floor() -> None:
    """The threshold is sharp and lands exactly where theory says: L - 1."""
    short = ofdm.simulate(6000, cfg(n_cp=3), 30.0, TAPS, True, 23).ber
    exact = ofdm.simulate(6000, cfg(n_cp=4), 30.0, TAPS, True, 23).ber
    assert exact < short / 5.0


@pytest.mark.parametrize("n_cp", [0, 1, 2])
def test_short_prefix_leaves_an_irreducible_floor(n_cp: int) -> None:
    """Raising the SNR by 20 dB must not clear it, which is what 'floor' means.

    Measured at 20 dB and 40 dB, n_cp = 0, 1, 2 improve by 1.1x, 1.2x and 2.7x
    - i.e. essentially not at all against a 100x change in noise power. At
    n_cp = 3 the single remaining tap of leakage is weak enough to improve 6x,
    so it is excluded here and covered by the threshold test above instead,
    which is the sharper statement anyway.
    """
    c = cfg(n_cp=n_cp)
    low = ofdm.simulate(4000, c, 20.0, TAPS, True, 24).ber
    high = ofdm.simulate(4000, c, 40.0, TAPS, True, 24).ber
    assert high > low / 3.0
    assert high > 1e-3


def test_the_prefix_threshold_is_sharp_at_the_delay_spread() -> None:
    """The finding: BER falls off a cliff exactly at n_cp = L - 1, not gradually.

    Measured at 30 dB on the 5-tap channel, the sweep over n_cp runs
    2.6e-2, 1.8e-2, 5.0e-3, 3.0e-3, then 3.4e-4 - more than an order of
    magnitude between n_cp = 3 and n_cp = 4, where theory says the circular
    convolution becomes exact.
    """
    L = TAPS.size
    below = ofdm.simulate(8000, cfg(n_cp=L - 2), 30.0, TAPS, True, 27).ber
    at = ofdm.simulate(8000, cfg(n_cp=L - 1), 30.0, TAPS, True, 27).ber
    assert at < below / 5.0


@pytest.mark.parametrize("n_cp", [0, 2, 3])
def test_no_prediction_is_offered_when_the_prefix_is_too_short(n_cp: int) -> None:
    """The prediction assumes no ISI, so it must decline rather than mislead."""
    assert ofdm.simulate(200, cfg(n_cp=n_cp), 20.0, TAPS, True, 25).predicted is None


def test_a_longer_prefix_than_needed_costs_energy() -> None:
    """Prefix beyond the delay spread is pure overhead, so BER gets slightly worse."""
    a = ofdm.simulate(12000, cfg(n_cp=4), 26.0, TAPS, True, 26).ber
    b = ofdm.simulate(12000, cfg(n_cp=24), 26.0, TAPS, True, 26).ber
    assert b > a


# ------------------------------------------------------ channel estimation
@pytest.mark.parametrize("method", ["dft", "linear"])
def test_estimation_is_exact_without_noise(method: str) -> None:
    c = cfg()
    rng = np.random.default_rng(30)
    bits = rng.integers(0, 2, size=20 * c.bits_per_block, dtype=np.uint8)
    tx = ofdm.modulate(bits, c)
    rx = np.convolve(tx, TAPS)[: tx.size]
    out, _ = ofdm.demodulate(rx, c, None, method)
    if method == "dft":
        assert np.array_equal(out, bits)
    else:
        # Linear interpolation has a modelling bias even with no noise at all.
        assert np.count_nonzero(out != bits) > 0


def test_dft_estimator_recovers_the_channel_taps() -> None:
    """It solves for the impulse response, so it should return the real one."""
    c = cfg()
    rng = np.random.default_rng(31)
    tx = ofdm.modulate(rng.integers(0, 2, size=4 * c.bits_per_block, dtype=np.uint8), c)
    rx = np.convolve(tx, TAPS)[: tx.size]
    blocks = rx.reshape(-1, c.samples_per_block)[:, c.n_cp:]
    grid = np.fft.fft(blocks, axis=1) / math.sqrt(c.n_sc)
    H_hat = ofdm.estimate_channel(grid, c, "dft")
    H_true = np.fft.fft(TAPS, c.n_sc)
    assert np.allclose(H_hat[0], H_true, atol=1e-10)


def test_linear_interpolation_has_an_snr_independent_error_floor() -> None:
    """The finding worth reporting: pilot spacing satisfies the sampling
    theorem here, but a straight line is not the matching reconstruction, so
    the residual is a bias and does not shrink with SNR."""
    c = cfg()
    lo = ofdm.simulate(9000, c, 10.0, TAPS, False, 32, "linear").ber
    hi = ofdm.simulate(9000, c, 26.0, TAPS, False, 32, "linear").ber
    assert hi > lo / 3.0
    assert hi > 1e-2


def test_dft_estimation_keeps_improving_with_snr() -> None:
    c = cfg()
    lo = ofdm.simulate(9000, c, 10.0, TAPS, False, 33, "dft").ber
    hi = ofdm.simulate(9000, c, 26.0, TAPS, False, 33, "dft").ber
    assert hi < lo / 5.0


@pytest.mark.parametrize("db", [10.0, 18.0, 26.0])
def test_dft_estimation_penalty_stays_bounded(db: float) -> None:
    """A fixed noise penalty from finite pilot SNR, not a growing bias."""
    c = cfg()
    perfect = ofdm.simulate(9000, c, db, TAPS, True, 34).ber
    est = ofdm.simulate(9000, c, db, TAPS, False, 34, "dft").ber
    assert est / max(perfect, 1e-12) < 3.0


def test_unknown_estimator_rejected() -> None:
    with pytest.raises(ValueError):
        ofdm.estimate_channel(np.ones((1, 64), complex), cfg(), "spline")


def test_equaliser_survives_a_deep_null_without_producing_nan() -> None:
    """An unguarded divide turns one faded subcarrier into inf, then NaN, then
    a poisoned block - and the slicer would still return bits."""
    c = cfg()
    rng = np.random.default_rng(35)
    bits = rng.integers(0, 2, size=10 * c.bits_per_block, dtype=np.uint8)
    tx = ofdm.modulate(bits, c)
    null_taps = np.array([1.0, -1.0], dtype=complex) / math.sqrt(2)  # null at DC
    rx = np.convolve(tx, null_taps)[: tx.size]
    out, syms = ofdm.demodulate(rx, c, null_taps)
    assert np.all(np.isfinite(syms))
    assert set(np.unique(out).tolist()) <= {0, 1}


@pytest.mark.parametrize("M", [4, 16])
def test_higher_order_modulation_is_worse_at_equal_ebn0(M: int) -> None:
    c4, cM = cfg(M=4), cfg(M=M)
    a = ofdm.simulate(6000, c4, 10.0, TAPS, True, 36).ber
    b = ofdm.simulate(6000, cM, 10.0, TAPS, True, 36).ber
    assert b >= a
