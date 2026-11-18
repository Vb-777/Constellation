"""Constellation: a digital communications link, built and measured against theory.

Single-carrier QAM with RRC shaping, timing and carrier recovery, plus an OFDM
chain with a cyclic prefix and pilot-based equalisation. Every performance
claim is a measured BER curve checked against a closed-form error probability.
"""
from . import channel, link, modulation, ofdm, pulse, sync, theory
from .link import Impairments, LinkConfig, ber_curve, simulate
from .modulation import QAM, ber, qam, random_bits
from .theory import awgn_ber_qpsk, awgn_ber_square_qam, qfunc, rayleigh_ber_qpsk

__version__ = "0.1.0"

__all__ = [
    "channel", "link", "modulation", "ofdm", "pulse", "sync", "theory",
    "LinkConfig", "Impairments", "simulate", "ber_curve",
    "QAM", "qam", "ber", "random_bits",
    "qfunc", "awgn_ber_qpsk", "awgn_ber_square_qam", "rayleigh_ber_qpsk",
]
