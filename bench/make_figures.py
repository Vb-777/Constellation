#!/usr/bin/env python3
"""Generate the figures in the README.

    python3 bench/make_figures.py

Writes docs/ber_curves.png, docs/constellations.png and docs/ofdm_prefix.png.
Every point is measured by the library, not sketched.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constellation import channel, link, ofdm, pulse, sync, theory
from constellation.link import Impairments, LinkConfig
from constellation.modulation import qam, random_bits

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)

INK = "#1b1b1f"
GRID = "#d9d9de"
SERIES = {4: "#2f6fb5", 16: "#c2561f", 64: "#3f7d53"}
FADE = "#8a4fa8"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def measured_point(M: int, db: float, target: int = 400) -> float:
    want = float(theory.awgn_ber_square_qam(M, db))
    n_bits = int(min(target / max(want, 1e-9), 3_000_000))
    r = link.simulate(max(n_bits, 80_000), LinkConfig(M=M),
                      Impairments(ebn0_db=db, timing_offset=0.37,
                                  freq_offset=1e-3, phase_offset=0.9), rng=5)
    return r.ber


def fig_ber() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid = np.linspace(0.0, 20.0, 400)

    names = {4: "QPSK", 16: "16-QAM", 64: "64-QAM"}
    for M in (4, 16, 64):
        ax.semilogy(grid, theory.awgn_ber_square_qam(M, grid),
                    color=SERIES[M], lw=1.6, label=f"{names[M]}, exact AWGN")
    ax.semilogy(grid, theory.rayleigh_ber_qpsk(grid), color=FADE, lw=1.6,
                ls="--", label="QPSK, flat Rayleigh")

    for M, pts in ((4, [2.0, 4.0, 6.0, 8.0]), (16, [6.0, 8.0, 10.0, 12.0])):
        ys = [measured_point(M, d) for d in pts]
        ax.semilogy(pts, ys, "o", ms=5.5, mfc="white", mew=1.6,
                    color=SERIES[M], label=f"{names[M]} measured, loops closed")

    ax.set_xlabel("$E_b/N_0$  (dB)")
    ax.set_ylabel("bit error rate")
    ax.set_title("Measured link against closed-form error probability", loc="left")
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-7, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    fig.text(0.5, 0.015,
             "markers: RRC $\\beta$=0.22, 0.37-sample timing offset, "
             "$10^{-3}$ cyc/sym carrier offset, both loops closed",
             ha="center", fontsize=7, color="#5a5a62")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "ber_curves.png")
    plt.close(fig)


def fig_constellations() -> None:
    """What the loops are actually for, in three panels."""
    M, db, sps = 16, 22.0, 8
    cfg = LinkConfig(M=M, sps=sps)
    a, h = cfg.alphabet, cfg.taps
    rng = np.random.default_rng(4)

    bits = random_bits(a.bits_per_symbol * 4000, rng)
    pre = sync.bpsk_preamble(cfg.preamble_len)
    syms = np.concatenate([pre, a.modulate(bits)])
    wave = pulse.apply_filter(pulse.upsample(syms, sps), h)
    wave = channel.fractional_delay(wave, 0.37)
    wave = channel.frequency_offset(wave, 1e-3 / sps, 0.9)
    wave = channel.awgn(wave, db, a.bits_per_symbol, rng)

    mf = pulse.apply_filter(wave, h)[max(h.size - 1 - sps, 0):]
    raw = mf[sps::sps][: syms.size]
    timed = sync.timing_recovery(mf, sps, cfg.timing_loop_bw).symbols
    off, _ = sync.find_preamble(timed, pre)
    timed = timed[off:]
    full = sync.carrier_recovery(timed, a, pre, cfg.carrier_loop_bw).symbols

    # A rotating carrier smears every panel into rings, which hides what the
    # timing loop fixed. Radial error - distance to the nearest constellation
    # ring - is invariant to rotation, so it isolates the timing and ISI part
    # and makes the middle panel's improvement a number rather than a squint.
    rings = np.unique(np.round(np.abs(a.points), 9))

    def radial_rms(d: np.ndarray) -> float:
        err = np.min(np.abs(np.abs(d)[:, None] - rings[None, :]), axis=1)
        return float(np.sqrt(np.mean(err**2)))

    panels = [
        (raw[pre.size:], "no recovery", "sampling instant and carrier both wrong"),
        (timed[pre.size:], "timing recovered", "radius sharp, carrier still turning"),
        (full[pre.size:], "timing + carrier", "the slicer can decide"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.3))
    for ax, (data, title, sub) in zip(axes, panels):
        d = data[:2500]
        ax.scatter(d.real, d.imag, s=1.4, color=SERIES[16], alpha=0.35,
                   linewidths=0)
        ax.scatter(a.points.real, a.points.imag, s=26, facecolors="none",
                   edgecolors=INK, linewidths=1.0, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=9)
        ax.set_xlabel("in-phase")
        ax.text(0.03, 0.03, f"radial RMS {radial_rms(d):.3f}",
                transform=ax.transAxes, fontsize=8, color="#5a5a62")
    axes[0].set_ylabel("quadrature")
    fig.suptitle(f"16-QAM at {db:g} dB $E_b/N_0$; open circles are the "
                 "transmitted constellation", x=0.012, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.015, 1, 0.94))
    fig.savefig(OUT / "constellations.png")
    plt.close(fig)


def fig_ofdm_prefix() -> None:
    taps = channel.EPA_TAPS
    L = taps.size
    counts = list(range(0, 25))
    ber, pred = [], []
    for ncp in counts:
        c = ofdm.OFDMConfig(64, ncp, 8, 4)
        r = ofdm.simulate(20_000, c, 30.0, taps, True, 23)
        ber.append(r.ber)
        pred.append(r.predicted if r.predicted is not None else np.nan)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(counts, ber, "o-", ms=4, color=SERIES[4], lw=1.4,
                label="measured")
    ax.semilogy(counts, pred, "s--", ms=4, color=SERIES[16], lw=1.4,
                label="per-subcarrier prediction")
    ax.axvline(L - 1, color=INK, lw=1.0, ls=":")
    ax.annotate(f"$N_{{cp}} = L-1 = {L - 1}$:\nlinear convolution\nbecomes circular",
                xy=(L - 1, 6e-4), xytext=(8.0, 6e-3), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("cyclic prefix length (samples)")
    ax.set_ylabel("bit error rate")
    ax.set_title("Cyclic prefix against a 5-tap channel, 30 dB, QPSK, perfect CSI",
                 loc="left", fontsize=10)
    ax.set_xlim(-0.8, 24.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.5, 0.015,
             "beyond the delay spread the prefix is pure overhead, "
             "so the curve turns back up",
             ha="center", fontsize=7, color="#5a5a62")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "ofdm_prefix.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_ber()
    print("wrote docs/ber_curves.png")
    fig_constellations()
    print("wrote docs/constellations.png")
    fig_ofdm_prefix()
    print("wrote docs/ofdm_prefix.png")
