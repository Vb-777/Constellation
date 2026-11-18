"""The analytic curves, checked against each other and against known values.

These are the tests that matter most, because every other test in the suite
measures something against `theory`. If the yardstick is wrong, everything
agrees with it and nothing is verified.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc

from constellation import theory

DB = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0])


@pytest.mark.parametrize("db", DB)
def test_qam4_enumeration_equals_the_closed_form(db: float) -> None:
    """The independent check on the whole enumeration machinery.

    For M = 4 the Gray-coded region integral must collapse to Q(sqrt(2 Eb/N0)).
    It agrees to 1e-13 relative all the way down to BER 1e-276 - but only after
    the region probabilities stopped being computed as a difference of CDFs.
    That formulation returned exactly half the right answer above 14 dB,
    because Phi(9) rounds to 1.0 in float64 and the upper tail cancelled to
    zero while the lower tail survived. This test is what caught it.
    """
    got = float(theory.awgn_ber_square_qam(4, db))
    want = float(theory.awgn_ber_qpsk(db))
    assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("db", [16.0, 20.0, 24.0, 28.0])
def test_far_tail_does_not_collapse_to_zero(db: float) -> None:
    """Guards the specific failure mode: silent underflow in the deep tail."""
    for M in (4, 16, 64):
        v = float(theory.awgn_ber_square_qam(M, db))
        assert v > 0.0, f"M={M} underflowed at {db} dB"
        assert math.isfinite(v)


@pytest.mark.parametrize("db", DB[DB >= 8.0])
def test_qam16_matches_the_standard_high_snr_approximation(db: float) -> None:
    """(3/8) erfc(sqrt(Eb/N0 / 2.5)) is the textbook 16-QAM approximation.

    It should agree closely once the second-order term is negligible, which is
    above roughly 8 dB. Agreement here is evidence the exact version is right;
    disagreement at low SNR, asserted separately below, is evidence it is
    genuinely exact rather than the same approximation in disguise.
    """
    got = float(theory.awgn_ber_square_qam(16, db))
    approx = 0.375 * float(erfc(math.sqrt(theory.db_to_linear(db) / 2.5)))
    assert got == pytest.approx(approx, rel=2e-4)


def test_exact_qam16_departs_from_the_approximation_at_low_snr() -> None:
    """At 0 dB the neglected term is worth 1.3%, so the two must not agree."""
    got = float(theory.awgn_ber_square_qam(16, 0.0))
    approx = 0.375 * float(erfc(math.sqrt(theory.db_to_linear(0.0) / 2.5)))
    assert abs(got / approx - 1.0) > 5e-3


@pytest.mark.parametrize("M", [4, 16, 64])
def test_ber_is_strictly_decreasing_in_snr(M: int) -> None:
    v = theory.awgn_ber_square_qam(M, DB)
    assert np.all(np.diff(v) < 0.0)


@pytest.mark.parametrize("db", [0.0, 6.0, 12.0])
def test_denser_constellations_are_worse_at_equal_ebn0(db: float) -> None:
    """More bits per symbol packs the points closer, so BER rises with M."""
    a = float(theory.awgn_ber_square_qam(4, db))
    b = float(theory.awgn_ber_square_qam(16, db))
    c = float(theory.awgn_ber_square_qam(64, db))
    assert a < b < c


@pytest.mark.parametrize("M", [4, 16, 64])
def test_ber_never_exceeds_ser_and_both_stay_in_range(M: int) -> None:
    ber = theory.awgn_ber_square_qam(M, DB)
    ser = np.atleast_1d(theory.awgn_ser_square_qam(M, DB))
    assert np.all((ber >= 0.0) & (ber <= 0.5))
    assert np.all((ser >= 0.0) & (ser <= 1.0))
    assert np.all(ber <= ser + 1e-15)


@pytest.mark.parametrize("M", [4, 16, 64])
def test_gray_labels_differ_in_one_bit_between_neighbours(M: int) -> None:
    """The defining property, and the reason BER is near SER/log2(M)."""
    n = int(round(math.sqrt(M)))
    g = theory.gray_labels(n)
    for a, b in zip(g, g[1:]):
        assert bin(int(a) ^ int(b)).count("1") == 1


@pytest.mark.parametrize("n", [2, 4, 8])
def test_gray_labels_are_a_permutation(n: int) -> None:
    assert sorted(theory.gray_labels(n).tolist()) == list(range(n))


def test_qfunc_against_known_values() -> None:
    assert float(theory.qfunc(0.0)) == pytest.approx(0.5)
    assert float(theory.qfunc(1.0)) == pytest.approx(0.158655253931, rel=1e-11)
    assert float(theory.qfunc(2.0)) == pytest.approx(0.022750131948, rel=1e-11)
    assert float(theory.qfunc(6.0)) == pytest.approx(9.865876450377e-10, rel=1e-9)


def test_qfunc_survives_the_far_tail() -> None:
    """Computing Q as 1 - cdf loses everything past x ~ 8; erfc does not."""
    assert float(theory.qfunc(20.0)) > 0.0
    assert float(theory.qfunc(20.0)) == pytest.approx(2.7536e-89, rel=1e-3)


@pytest.mark.parametrize("db", [20.0, 30.0, 40.0])
def test_rayleigh_decays_like_one_over_four_gamma(db: float) -> None:
    """The headline consequence of fading: polynomial, not exponential, decay.

    1/(4 gamma) is the high-SNR asymptote of the exact expression, so it is
    only asserted where it applies. At 0 dB the exact value is 0.146 against an
    asymptote of 0.25 - the asymptote is not merely loose there, it exceeds the
    true value, so testing it at low SNR would assert something false.
    """
    got = float(theory.rayleigh_ber_qpsk(db))
    g = theory.db_to_linear(db)
    assert got == pytest.approx(1.0 / (4.0 * g), rel=0.03)


@pytest.mark.parametrize("db", [-3.0, 0.0, 3.0])
def test_rayleigh_asymptote_overestimates_at_low_snr(db: float) -> None:
    """Which is why the asymptote is not used as the reference below 15 dB."""
    got = float(theory.rayleigh_ber_qpsk(db))
    assert got < 1.0 / (4.0 * theory.db_to_linear(db))
    assert got < 0.5


@pytest.mark.parametrize("db", [5.0, 10.0, 15.0, 20.0])
def test_rayleigh_is_far_worse_than_awgn(db: float) -> None:
    assert float(theory.rayleigh_ber_qpsk(db)) > float(theory.awgn_ber_qpsk(db))


def test_rayleigh_is_monotone_and_bounded() -> None:
    v = theory.rayleigh_ber_qpsk(DB)
    assert np.all(np.diff(v) < 0.0)
    assert np.all((v > 0.0) & (v <= 0.5))


@pytest.mark.parametrize("M", [2, 8, 32, 9, 0, -4])
def test_non_square_constellations_are_rejected(M: int) -> None:
    with pytest.raises(ValueError):
        theory.awgn_ber_square_qam(M, 10.0)


def test_scalar_in_scalar_out() -> None:
    v = theory.awgn_ber_square_qam(16, 10.0)
    assert np.ndim(v) == 0
    assert float(v) > 0.0
