"""Command line: measure a BER curve, run one link, or exercise the OFDM chain."""
from __future__ import annotations

import argparse

import numpy as np

from . import _native, link, ofdm, pulse, theory
from .channel import EPA_TAPS
from .link import Impairments, LinkConfig


def _snr_range(spec: str) -> np.ndarray:
    """Parse "start:stop:step" or a comma-separated list."""
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) == 2:
            parts.append(1.0)
        return np.arange(parts[0], parts[1] + 1e-9, parts[2])
    return np.array([float(x) for x in spec.split(",")])


def cmd_ber(a) -> int:
    db = _snr_range(a.snr)
    cfg = LinkConfig(M=a.M, sps=a.sps, beta=a.beta, span=a.span,
                     recover_timing=not a.ideal, recover_carrier=not a.ideal)
    imp = Impairments(timing_offset=0.0 if a.ideal else a.timing,
                      freq_offset=0.0 if a.ideal else a.freq)

    print(f"{a.M}-QAM, {a.sps} sps, RRC beta={a.beta} span={a.span}, "
          f"{'ideal sync' if a.ideal else 'loops closed'}")
    print(f"{'Eb/N0':>7} {'measured':>12} {'theory':>12} {'ratio':>7} "
          f"{'loss dB':>8} {'bits':>12}")

    ber, used = link.ber_curve(db, cfg, imp, target_errors=a.errors,
                               min_bits=a.min_bits, max_bits=a.max_bits, rng=a.seed)
    for d, b, n in zip(db, ber, used):
        t = float(theory.awgn_ber_square_qam(a.M, d))
        loss = _implementation_loss(a.M, d, b)
        ratio = f"{b / t:7.3f}" if t > 0 else "      -"
        loss_s = f"{loss:+8.2f}" if np.isfinite(loss) else "       -"
        print(f"{d:6.1f}dB {b:12.4e} {t:12.4e} {ratio} {loss_s} {n:12,d}")
    return 0


def _implementation_loss(M: int, db: float, measured: float) -> float:
    """dB of extra Eb/N0 the measured link needs to reach the analytic BER.

    The honest way to state an implementation penalty: not a BER ratio, which
    changes meaning with the slope of the curve, but the horizontal shift.

    Positive means worse than ideal. The real link at `db` performs like an
    ideal one at the SNR where theory gives the same BER, so the loss is `db`
    minus that SNR - not the other way round, which would report a penalty as
    a gain and make every impairment look beneficial.
    """
    if not np.isfinite(measured) or measured <= 0.0:
        return np.nan
    grid = np.arange(db - 12.0, db + 12.0, 0.005)
    curve = theory.awgn_ber_square_qam(M, grid)
    if measured > curve[0] or measured < curve[-1]:
        return np.nan
    equivalent = float(np.interp(-measured, -curve, grid))
    return db - equivalent


def cmd_link(a) -> int:
    cfg = LinkConfig(M=a.M, sps=a.sps, beta=a.beta, span=a.span)
    imp = Impairments(ebn0_db=a.snr, timing_offset=a.timing, freq_offset=a.freq,
                      phase_offset=a.phase)
    r = link.simulate(a.bits, cfg, imp, rng=a.seed)
    t = float(theory.awgn_ber_square_qam(a.M, a.snr))
    print(f"{a.M}-QAM at {a.snr:g} dB, {r.n_bits:,} bits")
    print(f"  bit errors          {r.n_errors:,} -> BER {r.ber:.4e}")
    print(f"  analytic BER        {t:.4e}   ratio {r.ber / t:.3f}" if t > 0 else "")
    print(f"  implementation loss {_implementation_loss(a.M, a.snr, r.ber):+.2f} dB")
    print(f"  EVM                 {r.evm_db:.2f} dB")
    print(f"  frame offset        {r.meta['frame_offset']} symbols "
          f"(peak ratio {r.meta['frame_ratio']:.1f})")
    if r.timing:
        print(f"  sampling phase      {r.timing.steady_state_tau():+.3f} samples "
              f"(injected {a.timing:+g}), jitter {r.timing.jitter():.3f}")
    if r.carrier:
        print(f"  frequency estimate  {r.carrier.f_coarse:+.3e} cyc/sym "
              f"(injected {a.freq:+g})")
    return 0


def cmd_ofdm(a) -> int:
    cfg = ofdm.OFDMConfig(a.n_sc, a.n_cp, a.pilots, a.M)
    taps = None if a.flat else EPA_TAPS
    print(f"OFDM {a.n_sc} subcarriers, CP {a.n_cp}, pilots every {a.pilots}, "
          f"{a.M}-QAM, {'flat' if a.flat else f'{EPA_TAPS.size}-tap'} channel")
    print(f"  {cfg.n_data} data subcarriers, overhead {ofdm.overhead_db(cfg):.2f} dB")
    print(f"{'Eb/N0':>7} {'perfect CSI':>12} {'predicted':>12} "
          f"{'DFT est':>12} {'linear est':>12}")
    for db in _snr_range(a.snr):
        p = ofdm.simulate(a.blocks, cfg, db, taps, True, a.seed)
        d = ofdm.simulate(a.blocks, cfg, db, taps, False, a.seed, "dft")
        l = ofdm.simulate(a.blocks, cfg, db, taps, False, a.seed, "linear")
        pred = f"{p.predicted:12.4e}" if p.predicted is not None else "    CP short"
        print(f"{db:6.1f}dB {p.ber:12.4e} {pred} {d.ber:12.4e} {l.ber:12.4e}")
    return 0


def cmd_info(a) -> int:
    print(f"C receiver: {'available' if _native.available() else 'NOT built'}")
    if not _native.available():
        print(f"  reason: {_native.why_unavailable()}")
    for beta in (0.0, 0.22, 0.35, 1.0):
        print(f"  RRC beta={beta:.2f} span=10 sps=8: "
              f"ISI {pulse.isi_residual(beta, 10, 8):.2e}, "
              f"floor {pulse.isi_floor_db(beta, 10, 8):.1f} dB, "
              f"bandwidth {pulse.excess_bandwidth(beta):.2f} x Rs")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="constellation", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("ber", help="measure a BER curve against theory")
    b.add_argument("--M", type=int, default=4, choices=[4, 16, 64])
    b.add_argument("--snr", default="2:10:2", help='"start:stop:step" or "4,8,12"')
    b.add_argument("--sps", type=int, default=8)
    b.add_argument("--beta", type=float, default=0.22)
    b.add_argument("--span", type=int, default=10)
    b.add_argument("--timing", type=float, default=0.37, help="sampling offset, samples")
    b.add_argument("--freq", type=float, default=1e-3, help="offset, cycles/symbol")
    b.add_argument("--ideal", action="store_true", help="no impairments, no loops")
    b.add_argument("--errors", type=int, default=200)
    b.add_argument("--min-bits", type=int, default=50_000)
    b.add_argument("--max-bits", type=int, default=4_000_000)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_ber)

    k = sub.add_parser("link", help="run one burst and report every receiver metric")
    k.add_argument("--M", type=int, default=16, choices=[4, 16, 64])
    k.add_argument("--snr", type=float, default=12.0)
    k.add_argument("--bits", type=int, default=400_000)
    k.add_argument("--sps", type=int, default=8)
    k.add_argument("--beta", type=float, default=0.22)
    k.add_argument("--span", type=int, default=10)
    k.add_argument("--timing", type=float, default=0.37)
    k.add_argument("--freq", type=float, default=1e-3)
    k.add_argument("--phase", type=float, default=0.9)
    k.add_argument("--seed", type=int, default=11)
    k.set_defaults(func=cmd_link)

    o = sub.add_parser("ofdm", help="OFDM over a frequency-selective channel")
    o.add_argument("--snr", default="6:18:4")
    o.add_argument("--n-sc", type=int, default=64)
    o.add_argument("--n-cp", type=int, default=16)
    o.add_argument("--pilots", type=int, default=8, help="pilot spacing")
    o.add_argument("--M", type=int, default=4, choices=[4, 16, 64])
    o.add_argument("--blocks", type=int, default=8000)
    o.add_argument("--flat", action="store_true", help="no multipath")
    o.add_argument("--seed", type=int, default=2)
    o.set_defaults(func=cmd_ofdm)

    i = sub.add_parser("info", help="build status and filter properties")
    i.set_defaults(func=cmd_info)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
