"""Mapper, slicer, and the normalisation that ties them to theory."""
from __future__ import annotations

import math

import numpy as np
import pytest

from constellation import theory
from constellation.modulation import QAM, ber, evm_db, qam, random_bits

MS = [4, 16, 64]


@pytest.mark.parametrize("M", MS)
def test_round_trip_is_lossless(M: int) -> None:
    a = qam(M)
    bits = random_bits(a.bits_per_symbol * 5000, 1)
    assert np.array_equal(a.demodulate(a.modulate(bits)), bits)


@pytest.mark.parametrize("M", MS)
def test_mean_symbol_energy_is_exactly_one(M: int) -> None:
    """Shared with theory.py. If this drifts, every curve shifts in dB silently."""
    assert qam(M).mean_energy == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("M", MS)
def test_empirical_energy_of_random_data_is_one(M: int) -> None:
    a = qam(M)
    s = a.modulate(random_bits(a.bits_per_symbol * 40000, 2))
    assert float(np.mean(np.abs(s) ** 2)) == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("M", MS)
def test_constellation_has_M_distinct_points(M: int) -> None:
    pts = qam(M).points
    assert pts.size == M
    assert np.unique(np.round(pts, 12)).size == M


@pytest.mark.parametrize("M", MS)
def test_nearest_neighbours_differ_in_exactly_one_bit(M: int) -> None:
    """Gray coding, verified through the mapper rather than on the labels.

    Every pair of points at the minimum distance must carry labels one bit
    apart. This is what makes a symbol error cost one bit, and it is asserted
    on the actual mapping because that is what the receiver uses.
    """
    a = qam(M)
    n = a.bits_per_symbol
    all_bits = np.array([[(i >> (n - 1 - b)) & 1 for b in range(n)]
                         for i in range(M)], dtype=np.uint8)
    pts = a.modulate(all_bits.ravel())
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = a.min_distance
    for i in range(M):
        for j in range(M):
            if i != j and abs(d[i, j] - dmin) < 1e-9:
                diff = int(np.count_nonzero(all_bits[i] != all_bits[j]))
                assert diff == 1, f"{i}<->{j} differ in {diff} bits"


@pytest.mark.parametrize("M", MS)
def test_slicer_is_the_maximum_likelihood_decision(M: int) -> None:
    """Per-dimension slicing must equal an exhaustive nearest-point search.

    The fast slicer exploits the uniform level spacing to avoid a search. That
    optimisation is only valid because square QAM decision regions are
    rectangles, so it is worth checking against the definition.
    """
    a = qam(M)
    rng = np.random.default_rng(3)
    noisy = a.modulate(random_bits(a.bits_per_symbol * 2000, rng)) + \
        (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)) * 0.25
    fast = a.demodulate(noisy)
    idx = np.argmin(np.abs(noisy[:, None] - a.points[None, :]), axis=1)
    brute = a.demodulate(a.points[idx])
    assert np.array_equal(fast, brute)


@pytest.mark.parametrize("M", MS)
def test_points_far_outside_the_constellation_clamp(M: int) -> None:
    """A deep fade or a bad equaliser produces huge samples; they must not wrap."""
    a = qam(M)
    wild = np.array([1e6 + 1e6j, -1e6 - 1e6j, 1e9 + 0j, np.nan * 0 + 0j])
    bits = a.demodulate(wild[:3])
    assert bits.size == 3 * a.bits_per_symbol
    assert set(np.unique(bits).tolist()) <= {0, 1}


@pytest.mark.parametrize("M", MS)
def test_noiseless_slicing_matches_theory_at_infinite_snr(M: int) -> None:
    a = qam(M)
    bits = random_bits(a.bits_per_symbol * 1000, 4)
    assert ber(bits, a.demodulate(a.modulate(bits))) == 0.0


@pytest.mark.parametrize("M", MS)
def test_min_distance_matches_the_level_spacing(M: int) -> None:
    a = qam(M)
    pts = a.points
    d = np.abs(pts[:, None] - pts[None, :])
    off_diag = d[d > 1e-12]
    assert float(np.min(off_diag)) == pytest.approx(a.min_distance, rel=1e-12)


def test_min_distance_shrinks_as_the_constellation_densifies() -> None:
    """At equal energy, 16-QAM points sit closer than QPSK's. Hence the BER gap."""
    assert qam(4).min_distance > qam(16).min_distance > qam(64).min_distance


@pytest.mark.parametrize("M", [2, 8, 32, 9, 1])
def test_invalid_order_rejected(M: int) -> None:
    with pytest.raises(ValueError):
        QAM(M)


@pytest.mark.parametrize("M", MS)
def test_bit_count_mismatch_is_an_error_not_a_truncation(M: int) -> None:
    """Silently dropping a partial symbol would shift every later bit."""
    a = qam(M)
    with pytest.raises(ValueError):
        a.modulate(np.ones(a.bits_per_symbol + 1, dtype=np.uint8))


def test_ber_counts_over_the_common_prefix() -> None:
    a = np.array([0, 1, 0, 1], np.uint8)
    b = np.array([0, 0, 0, 1, 1, 1], np.uint8)
    assert ber(a, b) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ber(np.empty(0, np.uint8), np.empty(0, np.uint8))


def test_evm_is_minus_infinity_when_perfect_and_rises_with_error() -> None:
    ref = qam(16).modulate(random_bits(400, 5))
    assert evm_db(ref, ref) == -np.inf
    rng = np.random.default_rng(6)
    small = ref + 0.01 * rng.standard_normal(ref.size)
    large = ref + 0.10 * rng.standard_normal(ref.size)
    assert evm_db(small, ref) < evm_db(large, ref)


def test_evm_matches_a_hand_computed_value() -> None:
    """A 10% RMS error vector on a unit-power reference is -20 dB."""
    ref = np.ones(1000, dtype=complex)
    rx = ref + 0.1
    assert evm_db(rx, ref) == pytest.approx(-20.0, abs=1e-9)


@pytest.mark.parametrize("M", MS)
def test_qam_cache_returns_the_same_object(M: int) -> None:
    assert qam(M) is qam(M)


@pytest.mark.parametrize("M", MS)
def test_bits_per_symbol_is_log2_M(M: int) -> None:
    assert qam(M).bits_per_symbol == int(round(math.log2(M)))


@pytest.mark.parametrize("M", MS)
def test_nearest_points_snaps_to_the_constellation(M: int) -> None:
    a = qam(M)
    rng = np.random.default_rng(7)
    noisy = a.modulate(random_bits(a.bits_per_symbol * 500, rng)) + \
        0.05 * rng.standard_normal(500)
    snapped = a.nearest_points(noisy)
    for s in snapped[:200]:
        assert np.min(np.abs(a.points - s)) < 1e-12


# Each pair is chosen so the analytic BER lands between about 1e-5 and 0.1 -
# measurable with a few million bits, and tight enough that a normalisation
# error of a few tenths of a dB would fail. Points far out in the tail are left
# to the analytic tests: at 12 dB QPSK sits near 9e-9, which would need some
# 4e9 bits for a stable estimate and would assert nothing a cheaper point does
# not already assert.
SNR_PAIRS = [(4, 2.0), (4, 4.0), (4, 6.0), (4, 8.0),
             (16, 4.0), (16, 8.0), (16, 10.0), (16, 12.0),
             (64, 8.0), (64, 12.0), (64, 15.0)]


@pytest.mark.parametrize("M,db", SNR_PAIRS)
def test_measured_ber_matches_exact_theory(M: int, db: float) -> None:
    """The headline check on the modulator: symbols plus noise, no filtering.

    The bit budget is set from the expected error count, so each point carries
    roughly the same confidence rather than the same cost.
    """
    from constellation.channel import awgn
    a = qam(M)
    rng = np.random.default_rng(11)
    want = float(theory.awgn_ber_square_qam(M, db))
    n_bits = min(int(400.0 / want), 6_000_000)
    n_sym = max(50_000, n_bits // a.bits_per_symbol)
    bits = random_bits(n_sym * a.bits_per_symbol, rng)
    got = ber(bits, a.demodulate(awgn(a.modulate(bits), db, a.bits_per_symbol, rng)))
    assert got == pytest.approx(want, rel=0.15)
