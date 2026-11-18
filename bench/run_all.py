#!/usr/bin/env python3
"""Every number in the README is produced by this script.

    python3 bench/run_all.py              # all six studies
    python3 bench/run_all.py ofdm         # one of them
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from constellation import _native, channel, link, ofdm, pulse, sync, theory
from constellation.link import Impairments, LinkConfig
from constellation.modulation import qam, random_bits

TAPS = channel.EPA_TAPS


def rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def implementation_loss(M: int, db: float, measured: float) -> float:
    """Extra Eb/N0 the real link needs to hit the analytic BER, in dB.

    Positive means worse than ideal: the real link at `db` performs like an
    ideal one at a lower SNR, and the loss is the gap.
    """
    if not np.isfinite(measured) or measured <= 0.0:
        return float("nan")
    grid = np.arange(db - 12.0, db + 12.0, 0.005)
    curve = theory.awgn_ber_square_qam(M, grid)
    if measured > curve[0] or measured < curve[-1]:
        return float("nan")
    return db - float(np.interp(-measured, -curve, grid))


# ------------------------------------------------------------------ 1. theory
def study_theory() -> None:
    rule("1. The analytic reference, and why it is computed exactly")
    db = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0])

    print("Gray QPSK: region enumeration vs the closed form Q(sqrt(2 Eb/N0))")
    print(f"{'Eb/N0':>7} {'enumerated':>16} {'closed form':>16} {'rel. diff':>11}")
    enum = theory.awgn_ber_square_qam(4, db)
    closed = theory.awgn_ber_qpsk(db)
    for d, a, b in zip(db, enum, closed):
        print(f"{d:6.1f}dB {a:16.8e} {b:16.8e} {abs(a - b) / b:11.2e}")
    print(f"  worst relative difference: {np.max(np.abs(enum - closed) / closed):.1e}")
    print("  Computed as a difference of CDFs this was a factor of 2 low above")
    print("  14 dB: Phi(9) rounds to exactly 1.0 in float64, so the upper tail")
    print("  cancelled to zero while the lower tail survived.")

    print("\n16-QAM: exact vs the textbook high-SNR approximation")
    print(f"{'Eb/N0':>7} {'exact':>16} {'(3/8)erfc(..)':>16} {'ratio':>9}")
    from scipy.special import erfc
    ex = theory.awgn_ber_square_qam(16, db)
    ap = 0.375 * erfc(np.sqrt(theory.db_to_linear(db) / 2.5))
    for d, a, b in zip(db, ex, ap):
        print(f"{d:6.1f}dB {a:16.8e} {b:16.8e} {a / b:9.5f}")
    print("  They agree to 5 decimals above 8 dB and part company at 0 dB,")
    print("  where the neglected second-order term is worth 1.3%.")

    print("\nAWGN against Rayleigh: exponential decay against polynomial")
    print(f"{'Eb/N0':>7} {'AWGN QPSK':>14} {'Rayleigh QPSK':>15} {'penalty':>12}")
    for d in (10.0, 15.0, 20.0, 30.0, 40.0):
        a = float(theory.awgn_ber_qpsk(d))
        r = float(theory.rayleigh_ber_qpsk(d))
        # Past about 25 dB the AWGN term underflows float64 entirely, which is
        # itself the point being made.
        pen = f"{r / a:11.3g}x" if a > 0.0 else "   unbounded"
        print(f"{d:6.1f}dB {a:14.4e} {r:15.4e} {pen}")
    print("  Averaging Q(sqrt(2g|h|^2)) over an exponential |h|^2 turns")
    print("  exponential decay into 1/(4g). One deep fade dominates the")
    print("  average, so 1e-5 needs about 40 dB instead of 10.")


# -------------------------------------------------------------- 2. pulse shape
def study_pulse() -> None:
    rule("2. Pulse shaping: what truncating a Nyquist filter costs")
    print(f"{'beta':>6} {'span':>6} {'peak distortion':>17} {'ISI floor dB':>14} "
          f"{'bandwidth':>11}")
    for beta in (0.0, 0.22, 0.35, 0.5, 1.0):
        for span in (6, 10, 20):
            print(f"{beta:6.2f} {span:6d} {pulse.isi_residual(beta, span, 8):17.4e} "
                  f"{pulse.isi_floor_db(beta, span, 8):14.1f} "
                  f"{pulse.excess_bandwidth(beta):10.2f}x")
    print("\n  The residual is the sum over non-zero symbol offsets, not the")
    print("  maximum: the largest single term always lands at the outermost")
    print("  offset, where the ideal cascade is zero and all that remains is")
    print("  the truncation step. The sum is what a symbol mid-burst sees.")

    print("\n  Span at beta = 0.35, finely:")
    print(f"{'span':>6} {'peak distortion':>17}")
    for span in range(4, 33, 2):
        print(f"{span:6d} {pulse.isi_residual(0.35, span, 8):17.4e}")
    print("  Not monotone. A truncated RRC is only approximately Nyquist and")
    print("  the residual depends on where the cut lands relative to the zero")
    print("  crossings of the tail, so doubling the span does not reliably")
    print("  halve the ISI. Hence 8 to 12 symbols in practice, not 32.")


# ----------------------------------------------------------- 3. the link curve
def study_link() -> None:
    rule("3. Single-carrier BER: measured against theory, loops closed")
    print("RRC beta=0.22 span=10, 8 sps, sampling offset 0.37 samples,")
    print("carrier offset 1e-3 cycles/symbol, initial phase 0.9 rad.\n")

    # Points are capped where the analytic BER is still around 1e-4. Beyond
    # that a reliable measurement needs tens of millions of bits - 10 dB QPSK
    # sits at 3.9e-6, so 100 errors would cost 26 million - and the comparison
    # would be dominated by the Monte-Carlo interval rather than by the
    # implementation loss it is meant to show.
    for M, pts in ((4, [2.0, 4.0, 6.0, 8.0]),
                   (16, [6.0, 8.0, 10.0, 12.0])):
        print(f"{M}-QAM")
        print(f"{'Eb/N0':>7} {'ideal sync':>12} {'loops closed':>13} "
              f"{'theory':>12} {'loss dB':>9} {'EVM dB':>8}")
        for db in pts:
            want = float(theory.awgn_ber_square_qam(M, db))
            n_bits = min(int(600.0 / want), 3_000_000)
            ideal_cfg = LinkConfig(M=M, recover_timing=False, recover_carrier=False)
            a = link.simulate(n_bits, ideal_cfg, Impairments(ebn0_db=db), rng=5)
            b = link.simulate(n_bits, LinkConfig(M=M),
                              Impairments(ebn0_db=db, timing_offset=0.37,
                                          freq_offset=1e-3, phase_offset=0.9), rng=5)
            print(f"{db:6.1f}dB {a.ber:12.4e} {b.ber:13.4e} {want:12.4e} "
                  f"{implementation_loss(M, db, b.ber):+9.2f} {b.evm_db:8.2f}")
        print()
    print("  The loss is quoted as a horizontal shift in dB, not a BER ratio:")
    print("  a ratio changes meaning with the slope of the curve, a dB does not.")


# --------------------------------------------------------------- 4. the loops
def study_sync() -> None:
    rule("4. Acquisition: what the loops actually recover")
    a = qam(4)
    sps, h = 8, pulse.rrc(0.22, 10, 8)

    print("Timing loop against an injected sampling offset (no noise)")
    print(f"{'injected':>10} {'recovered':>11} {'error':>9} {'jitter':>9}")
    for off in (-0.45, -0.2, 0.0, 0.2, 0.37, 0.45, 1.3):
        rng = np.random.default_rng(1)
        wave = pulse.apply_filter(
            pulse.upsample(a.modulate(random_bits(2 * 4000, rng)), sps), h)
        mf = pulse.apply_filter(channel.fractional_delay(wave, off), h)
        r = sync.timing_recovery(mf[h.size - 1 - sps:], sps, 0.005)
        t = r.steady_state_tau()
        print(f"{off:+10.3f} {t:+11.3f} {t - off:+9.3f} {r.jitter():9.4f}")
    print("  Jitter is non-zero with no noise at all: the Gardner error signal")
    print("  depends on the data, so random symbols drive the loop with a")
    print("  residual it cannot tell from a real timing error. Self-noise is")
    print("  irreducible at a given loop bandwidth.")

    print("\n  Blind to carrier phase, which is why it can run first:")
    rng = np.random.default_rng(2)
    wave = pulse.apply_filter(
        pulse.upsample(a.modulate(random_bits(2 * 3000, rng)), sps), h)
    wave = channel.fractional_delay(wave, 0.3)
    for ph in (0.0, 0.87, 2.4):
        mf = pulse.apply_filter(channel.phase_offset(wave, ph), h)
        t = sync.timing_recovery(mf[h.size - 1 - sps:], sps, 0.005).steady_state_tau()
        print(f"    carrier phase {ph:+.2f} rad -> sampling phase {t:+.4f}")

    print("\nFrequency estimate from the repeated preamble, 64 symbols")
    print(f"{'injected':>12} {'estimate':>13} {'error':>12}")
    p = sync.bpsk_preamble(64)
    for f in (0.0, 1e-4, 1e-3, 5e-3, -2e-3, 1e-2):
        got = sync.frequency_from_repetition(channel.frequency_offset(p, f, 0.6), 32)
        print(f"{f:+12.1e} {got:+13.3e} {got - f:+12.1e}")
    beyond = 1.0 / 32.0
    got = sync.frequency_from_repetition(channel.frequency_offset(p, beyond, 0.0), 32)
    print(f"  capture range is |f| < 1/(2*32) = {1/64:.4f}; at {beyond:.4f} it")
    print(f"  wraps and returns {got:+.4f}. Stated, not hidden.")

    print("\nWhy the preamble is repeated rather than merely known: variance.")
    print("One correlation at lag 32 beats 63 correlations at lag 1, because")
    print("the phase being measured has had 32 symbols to accumulate.")
    print("4000 trials per point, clean symbols, so this isolates noise:")
    print(f"{'Eb/N0':>7} {'lag-1 bias':>12} {'lag-1 std':>12} "
          f"{'rep bias':>12} {'rep std':>12} {'std ratio':>10}")
    rng = np.random.default_rng(10)
    for db in (0.0, 4.0, 8.0, 12.0, 20.0):
        e1, e2 = [], []
        for _ in range(4000):
            rx = channel.awgn(channel.frequency_offset(p, 1e-3, 0.6), db, 2, rng)
            e1.append(sync.coarse_frequency_estimate(rx, p) - 1e-3)
            e2.append(sync.frequency_from_repetition(rx, 32) - 1e-3)
        e1, e2 = np.array(e1), np.array(e2)
        print(f"{db:6.1f}dB {e1.mean():+12.2e} {e1.std():12.2e} "
              f"{e2.mean():+12.2e} {e2.std():12.2e} {e1.std() / e2.std():9.1f}x")
    print("  The advantage grows as SNR falls, which is where acquisition is")
    print("  hard. The std ratio is stable across seeds; read the bias columns")
    print("  as zero - each mean is within its own standard error, std/sqrt(4000),")
    print("  so neither estimator is measurably biased and only variance separates")
    print("  them. An earlier draft quoted a bias figure from one seed, which was")
    print("  a two-sigma fluctuation reported as a finding.")

    print("\n  And a correction worth recording. This change was first credited")
    print("  with fixing a 0.5 bit error rate, on the theory that payload ISI")
    print("  biased the lag-1 estimator in proportion to peak-to-average power.")
    print("  Measured on the pulse-shaped preamble, that bias is 6.5e-6 for")
    print("  QPSK, 4.9e-6 for 16-QAM and 2.7e-6 for 64-QAM - no trend with")
    print("  constellation order, and far too small to matter. The real cause")
    print("  was missing frame synchronisation, and both estimators work once")
    print("  the frame is aligned. Two things were changed in one sitting and")
    print("  the wrong one got the credit until it was measured.")

    print("\nRotational ambiguity: a decision-directed loop cannot see it")
    print(f"{'phase':>10} {'with preamble':>15} {'blind':>12}")
    for ph in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        al = qam(16)
        rng = np.random.default_rng(3)
        bits = random_bits(al.bits_per_symbol * 2000, rng)
        syms = np.concatenate([p, al.modulate(bits)])
        rx = channel.phase_offset(syms, ph)
        w = sync.carrier_recovery(rx, al, p, 0.01).symbols[p.size:]
        n = sync.carrier_recovery(rx, al, None, 0.01).symbols[p.size:]
        e1 = np.count_nonzero(al.demodulate(w) != bits) / bits.size
        e2 = np.count_nonzero(al.demodulate(n) != bits) / bits.size
        print(f"{ph:10.3f} {e1:15.4e} {e2:12.4e}")
    print("  A rotated square constellation is the same set of points, so the")
    print("  error signal is zero at all four lock points. Only known symbols")
    print("  break the tie; without them the BER is ~0.5 and every loop metric")
    print("  still looks healthy.")


# ------------------------------------------------------------------- 5. OFDM
def study_ofdm() -> None:
    rule("5. OFDM: the cyclic prefix, the overhead, and the estimator")
    cfg = ofdm.OFDMConfig(64, 16, 8, 4)
    print(f"64 subcarriers, CP 16, pilots every 8 -> {cfg.n_data} data carriers")
    print(f"Overhead (CP + pilots): {ofdm.overhead_db(cfg):.3f} dB "
          f"= 10 log10({cfg.samples_per_block}/{cfg.n_data})\n")

    print("Flat channel: must equal AWGN theory shifted by exactly that overhead")
    print(f"{'Eb/N0':>7} {'measured':>12} {'shifted theory':>15} {'ratio':>8}")
    for db in (4.0, 6.0, 8.0, 10.0):
        r = ofdm.simulate(30_000, cfg, db, None, True, 21)
        want = float(theory.awgn_ber_square_qam(4, db - ofdm.overhead_db(cfg)))
        print(f"{db:6.1f}dB {r.ber:12.4e} {want:15.4e} {r.ber / want:8.4f}")
    print("  Forgetting that term is what makes an OFDM curve appear to beat")
    print("  single-carrier QAM, which cannot happen.")

    print(f"\n{TAPS.size}-tap frequency-selective channel, perfect CSI, CP sufficient.")
    print("Prediction is the average over subcarriers of the AWGN BER at each")
    print("subcarrier's own SNR - exact, not an inequality.")
    print(f"{'Eb/N0':>7} {'measured':>12} {'predicted':>12} {'ratio':>8}")
    for db in (6.0, 10.0, 14.0, 18.0, 22.0):
        r = ofdm.simulate(20_000, cfg, db, TAPS, True, 20)
        print(f"{db:6.1f}dB {r.ber:12.4e} {r.predicted:12.4e} "
              f"{r.ber / r.predicted:8.4f}")

    print("\nCyclic prefix sweep at 30 dB. Channel is 5 taps, so it needs 4.")
    print(f"{'n_cp':>6} {'BER':>12} {'predicted':>12}  status")
    for ncp in (0, 1, 2, 3, 4, 6, 8, 16, 24):
        c = ofdm.OFDMConfig(64, ncp, 8, 4)
        r = ofdm.simulate(20_000, c, 30.0, TAPS, True, 23)
        pred = f"{r.predicted:12.4e}" if r.predicted is not None else "           -"
        status = "sufficient" if ncp >= TAPS.size - 1 else "TOO SHORT"
        print(f"{ncp:6d} {r.ber:12.4e} {pred}  {status}")
    print("  The threshold is sharp and lands exactly where the theory says:")
    print("  more than an order of magnitude between 3 and 4, because that is")
    print("  where linear convolution becomes circular. Past 4 the BER creeps")
    print("  back up - prefix beyond the delay spread is pure overhead.")

    print("\nShort prefix leaves a floor that SNR cannot clear")
    print(f"{'n_cp':>6} {'20 dB':>12} {'40 dB':>12} {'improvement':>13}")
    for ncp in (0, 1, 2, 3, 4):
        c = ofdm.OFDMConfig(64, ncp, 8, 4)
        lo = ofdm.simulate(12_000, c, 20.0, TAPS, True, 24).ber
        hi = ofdm.simulate(12_000, c, 40.0, TAPS, True, 24).ber
        imp = f"{lo / hi:12.1f}x" if hi > 0 else "    no errors"
        print(f"{ncp:6d} {lo:12.4e} {hi:12.4e} {imp}")

    print("\nChannel estimation: perfect CSI, transform domain, linear interpolation")
    print(f"{'Eb/N0':>7} {'perfect':>12} {'DFT':>12} {'linear':>12} "
          f"{'DFT pen':>9} {'lin pen':>9}")
    for db in (8.0, 12.0, 16.0, 20.0, 24.0):
        p = ofdm.simulate(20_000, cfg, db, TAPS, True, 34).ber
        d = ofdm.simulate(20_000, cfg, db, TAPS, False, 34, "dft").ber
        l = ofdm.simulate(20_000, cfg, db, TAPS, False, 34, "linear").ber
        print(f"{db:6.1f}dB {p:12.4e} {d:12.4e} {l:12.4e} "
              f"{d / p:8.2f}x {l / p:8.2f}x")
    print("  Pilot spacing of 8 satisfies the sampling theorem for a 5-tap")
    print("  channel (it must be below N/L = 12.8), so the linear estimator's")
    print("  floor is not undersampling - a straight line is simply not the")
    print("  reconstruction filter the sampling theorem calls for. The residual")
    print("  is a modelling bias and does not shrink with SNR. Solving for the")
    print("  taps instead and transforming costs a fixed noise penalty and no")
    print("  floor, which is what a real receiver does.")


# ----------------------------------------------------------- 6. the C fast path
def study_native() -> None:
    rule("6. The C receiver: where compiled code helps, and where it does not")
    if not _native.available():
        print(f"C library unavailable: {_native.why_unavailable()}")
        return

    a = qam(16)
    sps = 8
    rng = np.random.default_rng(0)
    bits = random_bits(a.bits_per_symbol * 40_000, rng)
    s = a.modulate(bits)
    h = pulse.rrc(0.22, 10, sps)
    wave = pulse.apply_filter(pulse.upsample(s, sps), h)
    aligned = np.convolve(wave, h, "full")[h.size - 1 - sps:]
    kp, ki = sync.loop_gains(0.005)
    n_sym = aligned.size // sps - 2

    py = sync.timing_recovery(aligned, sps, 0.005, backend="python")
    sym, tau, err = _native.gardner(aligned, sps, kp, ki, n_sym)
    print("Agreement with the Python reference:")
    print(f"  timing loop symbols   bit-identical: {np.array_equal(py.symbols, sym)}")
    print(f"  timing loop tau       bit-identical: {np.array_equal(py.tau, tau)}")
    print(f"  slicer, 40k symbols   mismatches:    "
          f"{np.count_nonzero(a.demodulate(s) != _native.demap_qam(s, a))}")
    print(f"  FIR filter            max abs diff:  "
          f"{np.max(np.abs(np.convolve(wave, h, 'full') - _native.mf_filter(wave, h))):.1e}")
    print("  Bit-identical, not merely close: the two must never disagree on a")
    print("  decision, or a BER measured with the fast path is not the BER of")
    print("  the reference receiver. Getting there needed nearbyint rather than")
    print("  llround in the slicer - they differ on exact ties, which have")
    print("  measure zero under noise and so would never show up in a BER.")

    def t(f, n=5):
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best * 1e3

    print(f"\n{'stage':>24} {'NumPy/Python':>14} {'C':>10} {'speedup':>9}")
    rows = [
        ("Gardner timing loop", lambda: sync.timing_recovery(aligned, sps, 0.005,
                                                             backend="python"),
         lambda: _native.gardner(aligned, sps, kp, ki, n_sym)),
        ("FIR matched filter", lambda: np.convolve(wave, h, "full"),
         lambda: _native.mf_filter(wave, h)),
        ("16-QAM slicer", lambda: a.demodulate(s),
         lambda: _native.demap_qam(s, a)),
        ("bit error count", lambda: int(np.count_nonzero(bits != bits)),
         lambda: _native.bit_errors(bits, bits)),
    ]
    for name, fnp, fc in rows:
        x, y = t(fnp), t(fc)
        print(f"{name:>24} {x:11.2f} ms {y:7.2f} ms {x / y:8.1f}x")
    print("\n  Only the first row is a real win, and the reason is structural:")
    print("  each strobe depends on the loop state the previous one left, so")
    print("  there is no array operation that computes them together and the")
    print("  Python version pays interpreter overhead per symbol.")
    print("  The others are already compiled inside NumPy. The filter is")
    print("  slightly slower in C than np.convolve, and the bit counter is")
    print("  several times slower because one ctypes call costs more than the")
    print("  whole vectorised comparison. Rewriting those in C would have been")
    print("  effort spent making the code worse.")


STUDIES = {"theory": study_theory, "pulse": study_pulse, "link": study_link,
           "sync": study_sync, "ofdm": study_ofdm, "native": study_native}

if __name__ == "__main__":
    which = sys.argv[1:] or list(STUDIES)
    for name in which:
        if name not in STUDIES:
            raise SystemExit(f"unknown study {name!r}; choose from {list(STUDIES)}")
        STUDIES[name]()
    print()
