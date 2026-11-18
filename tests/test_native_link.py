"""The C receiver against the Python reference, and the link end to end."""
from __future__ import annotations

import numpy as np
import pytest

from constellation import _native, channel, link, pulse, sync, theory
from constellation.link import Impairments, LinkConfig
from constellation.modulation import qam, random_bits

HAVE_C = _native.available()
needs_c = pytest.mark.skipif(not HAVE_C, reason=f"no C library: {_native.why_unavailable()}")


def _wave(M=16, n_sym=8000, sps=8, seed=0):
    a = qam(M)
    rng = np.random.default_rng(seed)
    bits = random_bits(a.bits_per_symbol * n_sym, rng)
    h = pulse.rrc(0.22, 10, sps)
    return a, bits, h, pulse.apply_filter(pulse.upsample(a.modulate(bits), sps), h)


# -------------------------------------------------------------- C vs Python
@needs_c
@pytest.mark.parametrize("sps", [2, 4, 8])
def test_c_timing_loop_is_bit_exact_with_the_python_reference(sps: int) -> None:
    """Not 'close'. Identical.

    Both implementations are the same recurrence in the same order with the
    same cubic interpolator, so every strobe, every sampling-phase estimate and
    every error sample must match to the last bit. Anything less would mean the
    two could disagree on a decision, and then a BER measured with the fast
    path would not be the BER of the reference receiver.
    """
    a, bits, h, wave = _wave(sps=sps, n_sym=4000, seed=1)
    aligned = pulse.apply_filter(wave, h)[max(h.size - 1 - sps, 0):]
    kp, ki = sync.loop_gains(0.005)
    n_sym = aligned.size // sps - 2

    py = sync.timing_recovery(aligned, sps, 0.005, backend="python")
    sym, tau, err = _native.gardner(aligned, sps, kp, ki, n_sym)

    assert np.array_equal(py.symbols, sym)
    assert np.array_equal(py.tau, tau)
    assert np.array_equal(py.error, err)


@needs_c
@pytest.mark.parametrize("M", [4, 16, 64])
def test_c_slicer_agrees_on_every_bit(M: int) -> None:
    a = qam(M)
    rng = np.random.default_rng(2)
    noisy = a.modulate(random_bits(a.bits_per_symbol * 20000, rng)) + \
        0.2 * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000))
    assert np.array_equal(a.demodulate(noisy), _native.demap_qam(noisy, a))


@needs_c
@pytest.mark.parametrize("M", [4, 16, 64])
def test_c_slicer_agrees_on_pathological_inputs(M: int) -> None:
    """Clamping behaviour must match too, or a deep fade decodes differently."""
    a = qam(M)
    wild = np.array([0j, 1e9 + 0j, -1e9 + 0j, 1e-30 + 1e-30j, -3 - 3j, 5 + 5j])
    assert np.array_equal(a.demodulate(wild), _native.demap_qam(wild, a))


@needs_c
def test_c_filter_matches_numpy_convolution() -> None:
    _, _, h, wave = _wave(n_sym=3000, seed=3)
    assert np.allclose(_native.mf_filter(wave, h), np.convolve(wave, h, "full"),
                       atol=1e-12)


@needs_c
def test_c_bit_error_count_matches_numpy() -> None:
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, size=50000, dtype=np.uint8)
    b = rng.integers(0, 2, size=50000, dtype=np.uint8)
    assert _native.bit_errors(a, b) == int(np.count_nonzero(a != b))


@needs_c
def test_c_wrappers_accept_non_contiguous_input() -> None:
    """A sliced array has the right dtype and the wrong strides; taking a raw
    pointer to it would read the wrong memory and return plausible nonsense."""
    a, _, h, wave = _wave(n_sym=2000, seed=5)
    strided = wave[::2]
    assert np.allclose(_native.mf_filter(strided, h),
                       np.convolve(np.ascontiguousarray(strided), h, "full"),
                       atol=1e-12)


@needs_c
def test_selecting_the_c_backend_explicitly_works() -> None:
    a, _, h, wave = _wave(n_sym=2000, seed=6)
    aligned = pulse.apply_filter(wave, h)[h.size - 1 - 8:]
    r = sync.timing_recovery(aligned, 8, 0.005, backend="c")
    assert r.symbols.size > 1000


def test_fallback_path_is_used_when_c_is_absent(monkeypatch) -> None:
    """The package must work unbuilt; C is an optimisation, not a dependency."""
    monkeypatch.setattr(_native, "load", lambda: None)
    a, _, h, wave = _wave(n_sym=500, seed=7)
    assert np.allclose(_native.mf_filter(wave, h), np.convolve(wave, h, "full"))
    with pytest.raises(RuntimeError):
        _native.gardner(wave, 8, 0.1, 0.01, 10)


# ------------------------------------------------------------ link end to end
@pytest.mark.parametrize("M", [4, 16])
def test_unimpaired_link_is_lossless(M: int) -> None:
    """Every impairment off must give exactly zero errors. If this fails, some
    stage is lossy on its own and every later measurement is contaminated."""
    r = link.simulate(40_000, LinkConfig(M=M), Impairments(), rng=0)
    assert r.n_errors == 0


@pytest.mark.parametrize("M,db", [(4, 4.0), (4, 6.0), (4, 8.0),
                                  (16, 8.0), (16, 10.0), (16, 12.0)])
def test_full_chain_tracks_theory_within_the_implementation_loss(M: int, db: float) -> None:
    """All impairments on and both loops closed. The measured BER must sit
    within a realistic implementation loss of the analytic curve - above it,
    because every real impairment costs something, and not far above it.

    SNR points are chosen so a few hundred errors accumulate in the bit budget.
    Pushing further into the tail would widen the confidence interval past the
    bound being asserted, and the test would then be measuring the random seed.
    """
    cfg = LinkConfig(M=M)
    imp = Impairments(ebn0_db=db, timing_offset=0.37, freq_offset=1e-3, phase_offset=0.9)
    want = float(theory.awgn_ber_square_qam(M, db))
    n_bits = min(int(400.0 / want), 3_000_000)
    r = link.simulate(max(n_bits, 100_000), cfg, imp, rng=11)
    assert 0.8 * want < r.ber < 2.0 * want


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.49, -0.3])
def test_timing_offset_alone_is_recovered(offset: float) -> None:
    r = link.simulate(40_000, LinkConfig(M=4),
                      Impairments(timing_offset=offset), rng=1)
    assert r.n_errors == 0


@pytest.mark.parametrize("phase", [0.0, 1.0, 2.5, -3.0])
def test_phase_offset_alone_is_recovered(phase: float) -> None:
    r = link.simulate(40_000, LinkConfig(M=16),
                      Impairments(phase_offset=phase), rng=2)
    assert r.n_errors == 0


@pytest.mark.parametrize("f", [0.0, 5e-4, 1e-3, -1e-3])
def test_frequency_offset_alone_is_recovered(f: float) -> None:
    r = link.simulate(40_000, LinkConfig(M=4), Impairments(freq_offset=f), rng=3)
    assert r.n_errors == 0


def test_frame_offset_is_found_and_reported() -> None:
    r = link.simulate(20_000, LinkConfig(M=4), Impairments(timing_offset=0.3), rng=4)
    assert r.meta["frame_ratio"] > 3.0


def test_evm_degrades_monotonically_with_noise() -> None:
    """EVM moves while BER is still exactly zero, which is why it is reported."""
    cfg = LinkConfig(M=16)
    vals = [link.simulate(20_000, cfg, Impairments(ebn0_db=db), rng=5).evm_db
            for db in (14.0, 18.0, 22.0, 26.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_disabling_recovery_still_works_when_there_is_nothing_to_recover() -> None:
    cfg = LinkConfig(M=4, recover_timing=False, recover_carrier=False)
    r = link.simulate(20_000, cfg, Impairments(), rng=6)
    assert r.n_errors == 0


@pytest.mark.parametrize("db", [6.0, 10.0, 14.0, 18.0, 22.0])
def test_memoryless_rayleigh_matches_the_closed_form(db: float) -> None:
    """The right place to validate the fading curve: the memoryless channel.

    One symbol, one gain, no filter memory - exactly what
    `theory.rayleigh_ber_qpsk` derives. Agreement is within 1%, tight enough
    that the tolerance here is set by the Monte-Carlo interval rather than by
    any modelling slack.
    """
    r = link.simulate_memoryless(6_000_000, 4, db, rayleigh=True, rng=40)
    assert r.ber == pytest.approx(float(theory.rayleigh_ber_qpsk(db)), rel=0.05)
    assert r.ber > 5.0 * float(theory.awgn_ber_qpsk(db))


@pytest.mark.parametrize("db", [2.0, 4.0, 6.0, 8.0])
def test_memoryless_awgn_matches_the_closed_form(db: float) -> None:
    """Capped at 8 dB, where QPSK is near 1.9e-4. At 12 dB it is 9e-9, which
    3e6 bits cannot measure - the expected error count there is 0.03."""
    r = link.simulate_memoryless(3_000_000, 4, db, rayleigh=False, rng=41)
    assert r.ber == pytest.approx(float(theory.awgn_ber_qpsk(db)), rel=0.12)


def test_pulse_shaped_fading_is_not_the_memoryless_model() -> None:
    """A measured statement of why the previous test uses the memoryless path.

    A fade redrawn every 40 symbols leaves the matched filter - which spans 10
    symbols - straddling two different gains for a quarter of all symbols. No
    single gain undoes that, so the waveform link sits well above the analytic
    curve at high SNR even with perfect channel knowledge. Lengthening the fade
    block fixes it from one side and starves the average of independent fades
    from the other; 200 symbols is near the sweet spot, at 1.09x.

    The point is that the gap is a modelling mismatch, not a receiver defect,
    so it is asserted rather than tuned away.
    """
    cfg = LinkConfig(M=4, recover_timing=False, recover_carrier=False,
                     preamble_len=0)
    db = 18.0
    want = float(theory.rayleigh_ber_qpsk(db))

    def run(block: int) -> float:
        errs = bits = 0
        for seed in range(8):
            r = link.simulate(200_000, cfg,
                              Impairments(ebn0_db=db, rayleigh=True,
                                          rayleigh_block=block),
                              rng=200 + seed)
            errs += r.n_errors
            bits += r.n_bits
        return errs / bits

    assert run(40) > 1.7 * want           # boundary contamination dominates
    assert run(200) == pytest.approx(want, rel=0.3)


def test_ber_curve_is_monotone_and_spends_bits_where_needed() -> None:
    """Stopping on an error count, not a bit count, is what keeps the high-SNR
    end meaningful instead of ragged."""
    cfg = LinkConfig(M=4, recover_timing=False, recover_carrier=False)
    db = np.array([4.0, 6.0, 8.0])
    ber, used = link.ber_curve(db, cfg, target_errors=120, min_bits=20_000,
                               max_bits=800_000, rng=8)
    assert np.all(np.diff(ber) < 0.0)
    assert used[-1] > used[0]


def test_zero_payload_is_rejected() -> None:
    with pytest.raises(ValueError):
        link.simulate(0, LinkConfig(M=4), Impairments(), rng=9)


def test_sub_symbol_echo_is_not_intersymbol_interference() -> None:
    """Selectivity is relative to the symbol rate, not a property of a tap list.

    `EPA_TAPS` spans 5 samples. At 8 samples per symbol that is 0.5 of a symbol
    period, so it is a sub-symbol echo: it distorts the pulse and rotates the
    phase, and the matched filter plus the carrier loop absorb most of it. The
    same taps in the OFDM chain, where the samples *are* the transform grid,
    span 5 of 64 and are genuinely frequency-selective across the band.

    Getting this backwards is how a single-carrier equaliser gets blamed for a
    channel that never needed one.
    """
    cfg = LinkConfig(M=4)
    bad = link.simulate(60_000, cfg,
                        Impairments(ebn0_db=20.0, multipath_taps=channel.EPA_TAPS),
                        rng=10)
    assert bad.ber < 1e-4


def test_symbol_spaced_echo_destroys_an_unequalised_link() -> None:
    """A delayed copy a full symbol late is real ISI, and nothing here fixes it.

    This is the motivation for the OFDM chain: the single-carrier receiver has
    no equaliser, so a symbol-spaced echo at -3 dB is fatal however much SNR is
    available.
    """
    sps = LinkConfig().sps
    taps = np.zeros(2 * sps + 1, dtype=complex)
    taps[0] = 1.0
    taps[sps] = 0.7            # one full symbol late
    taps[2 * sps] = 0.4        # two symbols late
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2))
    r = link.simulate(60_000, LinkConfig(M=4),
                      Impairments(ebn0_db=30.0, multipath_taps=taps), rng=11)
    assert r.ber > 1e-2
