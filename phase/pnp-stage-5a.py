"""
Stage-5A plug-and-play phase-error SDP:
joint trusted beam splitter + destructive photon-number analyzer (PNA).

This file implements the Stage-5A chain written in main.tex:

    physical BS + PNA
        -> H_m(N_plus)
        -> delta(N_plus)
        -> epsilon_5
        -> beta <= epsilon_5
        -> finite accepted-Fock Gram SDP.

There is no separate trusted attenuator in Stage 5A.  A trusted beam splitter
with signal-port transmissivity tau_s sends the complementary fraction
1-tau_s to the destructive monitor.  The secret phase encoding is applied
only after the monitor decision.

One-arm notation
----------------
M : incoming photon number
C : monitor-port photon number
N : surviving signal photon number

For an ideal lossless beam splitter,

    M = C + N,

and

    N | M=m ~ Binomial(m, tau_s).

A general phase-insensitive monitor is described by

    a_c = Pr(A_mon | C=c).

The exact monitor-conditioned surviving photon distribution is

    q_n^acc
      = (1/P_mon) sum_{m>=n}
          p_m Bin(m,n;tau_s) a_{m-n},

and the exact phase overlap is

    Lambda_acc(theta)
      = sum_n q_n^acc exp(i theta n)
      = [sum_m p_m F_m(theta)] / [sum_m p_m A_m],

where

    F_m(theta)
      = sum_c a_c Bin(m,c;1-tau_s) exp(i theta (m-c)),

and A_m = F_m(0).

The final Stage-5A SDP does NOT retain the unknown incoming distribution p_m.
Instead, the physical monitor model is used to certify

    Pr[
        N_A > N_A_plus or N_B > N_B_plus
        | both monitors accept
    ] <= epsilon_5.

The accepted surviving distribution is then decomposed into a finite good
component plus arbitrary bad mass beta <= epsilon_5.  The good component is
represented exactly on the rectangular Fock support

    0 <= n_A <= N_A_plus,
    0 <= n_B <= N_B_plus,

and the bad component contributes at most beta in modulus to every overlap.
This is an outer relaxation of the exact PNA-BS instrument.

Ideal architecture benchmark
----------------------------
For the special hard upper-threshold response

    a_c = 1[c <= c_max],

the one-arm tail quantity

    delta(N_plus)
      = sup_m Pr(A_mon and N > N_plus | M=m)

has the exact maximizer

    m_star = c_max + N_plus + 1,

with

    delta_ideal(N_plus)
      = Pr[Binomial(m_star, tau_s) > N_plus].

This closed form is an idealized architecture sanity check only.  For a
realistic soft response a_c, delta must be bounded separately.

Important rate convention
-------------------------
The Gram SDP is conditioned on monitor acceptance.  Its observed central-node
statistics are therefore

    P(z | s, A_mon),

and the key rate returned by the SDP is per monitor-accepted round:

    R_acc
      = P_pass|mon
        [1 - h2(e_ph) - f_EC h2(e_bit)].

A rate per original incoming round additionally requires the observed physical

    P_mon = Pr(A_mon),

and is

    R_original = P_mon * R_acc.

This file contains an optional synthetic accepted-ensemble channel model for
numerical experiments.  That model is not a derivation of P_mon and is not a
physical detector model.

No solve is performed on import.  The guarded main() runs only when this file
is explicitly executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import cvxpy as cp
import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Protocol / synthetic accepted-ensemble parameters

M_DEFAULT = 4
F_EC = 1.15
ETA_DET = 0.85
XI = 0.2
E_ALI = 0.015
P_DC = 5e-8
LOSS_ONLY = False

# Tiny numerical allowance added to SOC overlap constraints.  This is not a
# physical leakage parameter and must remain much smaller than epsilon_5.
SLACK_FLOOR = 1e-12

Z_P, Z_M, Z_F = 0, 1, 2
Z_ALL = (Z_P, Z_M, Z_F)

# Four bit-pair phase offsets inside each basis.
ENC_PHASES = (
    (0.0, 0.0),
    (0.0, np.pi),
    (np.pi, 0.0),
    (np.pi, np.pi),
)


# ---------------------------------------------------------------------------
# Data containers


@dataclass(frozen=True)
class IdealTailResult:
    tau_s: float
    c_max: int
    n_plus: int
    m_star: int
    delta: float


@dataclass(frozen=True)
class IdealTwoArmInterface:
    delta_a: float
    delta_b: float
    epsilon_5: float
    p_mon_both: float
    nA_plus: int
    nB_plus: int


@dataclass
class Stage5AResult:
    e_ph_upper: float
    e_bit: float
    p_pass_mon: float
    r_acc: float
    r_original: float | None
    status: str
    q_good: np.ndarray | None
    support: np.ndarray
    diagnostics: dict


# ---------------------------------------------------------------------------
# Validation / elementary helpers


def _check_tau(tau_s: float) -> None:
    if not (0.0 < tau_s < 1.0):
        raise ValueError("tau_s must lie strictly between 0 and 1.")


def _check_nonnegative_int(name: str, value: int) -> None:
    if int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _check_probability(name: str, value: float, *, positive: bool = False) -> None:
    if positive:
        ok = 0.0 < value <= 1.0
        interval = "(0,1]"
    else:
        ok = 0.0 <= value <= 1.0
        interval = "[0,1]"
    if not ok:
        raise ValueError(f"{name} must lie in {interval}.")


def h2(e: float) -> float:
    """Binary entropy, with the endpoint convention h2(0)=h2(1)=0."""
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return float(-e * np.log2(e) - (1.0 - e) * np.log2(1.0 - e))


def state_phases(M: int) -> list[tuple[float, float]]:
    """
    Signal-label phases.

    For basis x, phi_x = x*pi/M.  The four labels in that basis correspond to
    bit pairs 00, 01, 10, 11 through the offsets in ENC_PHASES.
    """
    if M < 1:
        raise ValueError("M must be positive.")
    out = []
    for x in range(M):
        phi = np.pi * x / M
        for dA, dB in ENC_PHASES:
            out.append((phi + dA, phi + dB))
    return out


# ---------------------------------------------------------------------------
# Optional synthetic central-node statistics conditioned on monitor acceptance


def synthetic_channel(n_bar: float, distance_km: float) -> tuple[float, float]:
    """
    Synthetic accepted-ensemble central-node model used only for exploration.

    n_bar is the nominal total honest signal brightness AFTER the local
    monitor/beam-splitter stage, conditioned on monitor acceptance.

    Returns
    -------
    P_pass_mon : float
        Synthetic central pass probability conditioned on A_mon.
    e_bit : float
        Synthetic accepted-ensemble bit-error rate.

    This function does NOT determine P_mon and does NOT model the PNA.
    """
    if n_bar < 0.0:
        raise ValueError("n_bar must be nonnegative.")
    if distance_km < 0.0:
        raise ValueError("distance_km must be nonnegative.")

    eta = ETA_DET * 10 ** (-XI * distance_km / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI

    p_sig = 1.0 - np.exp(-eta * n_bar)
    p_pass = p_sig + p_dc

    if p_pass <= 0.0:
        return 0.0, 0.0

    e_bit = (e_ali * p_sig + 0.5 * p_dc) / p_pass
    return float(p_pass), float(e_bit)


def synthetic_announcement_probabilities(
    M: int,
    n_bar: float,
    distance_km: float,
) -> tuple[dict[int, np.ndarray], float, float]:
    """
    Build P(z | s, A_mon) for z in {+,-,fail} using the same synthetic model
    used in Stages 3 and 4.

    This is a convenience for numerical testing of the Stage-5A SDP, not a
    physical source-monitor model.
    """
    ns = 4 * M
    p_pass, e_bit = synthetic_channel(n_bar, distance_km)

    p_plus = np.zeros(ns)
    p_minus = np.zeros(ns)

    for s in range(ns):
        hit = (1.0 - e_bit) * p_pass
        miss = e_bit * p_pass
        if (s % 4) in (0, 3):
            p_plus[s], p_minus[s] = hit, miss
        else:
            p_plus[s], p_minus[s] = miss, hit

    p_fail = 1.0 - p_plus - p_minus
    if np.any(p_fail < -1e-12):
        raise ValueError("Synthetic probabilities are inconsistent.")

    p_fail = np.maximum(p_fail, 0.0)
    prob_of = {Z_P: p_plus, Z_M: p_minus, Z_F: p_fail}
    return prob_of, p_pass, e_bit


# ---------------------------------------------------------------------------
# Monitor response models


def ideal_upper_response(c: int, c_max: int) -> float:
    """Ideal hard upper threshold: a_c = 1[c <= c_max]."""
    _check_nonnegative_int("c", c)
    _check_nonnegative_int("c_max", c_max)
    return 1.0 if c <= c_max else 0.0


# ---------------------------------------------------------------------------
# Exact fixed-m beam-splitter / PNA quantities


def ideal_H_m(m: int, n_plus: int, tau_s: float, c_max: int) -> float:
    """
    Exact ideal-threshold bad-event probability

        H_m(N_plus)
          = Pr(C <= c_max, N > N_plus | M=m).

    Since N ~ Binomial(m,tau_s), the event is

        N >= max(N_plus + 1, m-c_max).
    """
    _check_tau(tau_s)
    for name, value in (("m", m), ("n_plus", n_plus), ("c_max", c_max)):
        _check_nonnegative_int(name, value)

    if m <= n_plus:
        return 0.0

    lower = max(n_plus + 1, m - c_max)
    if lower > m:
        return 0.0

    return float(binom.sf(lower - 1, m, tau_s))


def general_H_m(
    m: int,
    n_plus: int,
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> float:
    """
    Exact finite sum for a general phase-insensitive monitor response:

        H_m(N_plus)
          = sum_{n=N_plus+1}^m Bin(m,n;tau_s) a_{m-n}.
    """
    _check_tau(tau_s)
    _check_nonnegative_int("m", m)
    _check_nonnegative_int("n_plus", n_plus)

    if m <= n_plus:
        return 0.0

    n = np.arange(n_plus + 1, m + 1, dtype=int)
    c = m - n
    a = np.asarray([float(accept_prob(int(ci))) for ci in c], dtype=float)

    if np.any((a < 0.0) | (a > 1.0)):
        raise ValueError("accept_prob(c) must lie in [0,1].")

    return float(np.sum(binom.pmf(n, m, tau_s) * a))


def F_m(
    m: int,
    theta: float,
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> complex:
    """
    Exact fixed-input acceptance-weighted phase kernel:

        F_m(theta)
          = E[a_C exp(i theta N) | M=m]
          = sum_c a_c Bin(m,c;1-tau_s) exp(i theta (m-c)).

    F_m(0) = A_m = Pr(A_mon | M=m).
    """
    _check_tau(tau_s)
    _check_nonnegative_int("m", m)

    c = np.arange(m + 1, dtype=int)
    n = m - c
    a = np.asarray([float(accept_prob(int(ci))) for ci in c], dtype=float)

    if np.any((a < 0.0) | (a > 1.0)):
        raise ValueError("accept_prob(c) must lie in [0,1].")

    weights = binom.pmf(c, m, 1.0 - tau_s)
    return complex(np.sum(a * weights * np.exp(1j * theta * n)))


# ---------------------------------------------------------------------------
# Exact accepted distribution for a finite input distribution


def accepted_signal_distribution(
    p_m: Sequence[float],
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> tuple[np.ndarray, float]:
    """
    Exact accepted surviving photon-number distribution for a finite supplied
    input distribution p_m, m=0,...,M_max.

    This is exact for the supplied finite p_m.  It is NOT a truncation
    certificate for an unknown infinite-tail source.
    """
    _check_tau(tau_s)

    p = np.asarray(p_m, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_m must be a nonempty one-dimensional sequence.")
    if np.any(p < 0.0):
        raise ValueError("p_m must be nonnegative.")
    if not np.isclose(np.sum(p), 1.0, atol=1e-12):
        raise ValueError("p_m must sum to 1.")

    m_max = len(p) - 1
    q_tilde = np.zeros(m_max + 1, dtype=float)

    for m, pm in enumerate(p):
        if pm == 0.0:
            continue

        n = np.arange(m + 1, dtype=int)
        c = m - n
        a = np.asarray([float(accept_prob(int(ci))) for ci in c], dtype=float)

        if np.any((a < 0.0) | (a > 1.0)):
            raise ValueError("accept_prob(c) must lie in [0,1].")

        q_tilde[: m + 1] += pm * binom.pmf(n, m, tau_s) * a

    p_mon = float(np.sum(q_tilde))
    if p_mon <= 0.0:
        raise ValueError("Monitor acceptance probability is zero.")

    return q_tilde / p_mon, p_mon


def accepted_overlap_from_q(q_acc: Sequence[float], theta: float) -> complex:
    """
    Exact phase overlap

        Lambda_acc(theta) = sum_n q_n^acc exp(i theta n).
    """
    q = np.asarray(q_acc, dtype=float)
    if q.ndim != 1 or len(q) == 0:
        raise ValueError("q_acc must be a nonempty one-dimensional sequence.")
    if np.any(q < -1e-15):
        raise ValueError("q_acc must be nonnegative.")
    if not np.isclose(np.sum(q), 1.0, atol=1e-10):
        raise ValueError("q_acc must sum to 1.")

    n = np.arange(len(q), dtype=float)
    return complex(np.sum(q * np.exp(1j * theta * n)))


def direct_overlap_from_p(
    p_m: Sequence[float],
    theta: float,
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> tuple[complex, float]:
    """
    Evaluate Eq. (127) directly for a finite supplied p_m:

        Lambda_acc(theta)
          = sum_m p_m F_m(theta) / sum_m p_m F_m(0).

    Returns (Lambda_acc, P_mon).
    """
    p = np.asarray(p_m, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_m must be a nonempty one-dimensional sequence.")
    if np.any(p < 0.0):
        raise ValueError("p_m must be nonnegative.")
    if not np.isclose(np.sum(p), 1.0, atol=1e-12):
        raise ValueError("p_m must sum to 1.")

    f_theta = np.asarray(
        [F_m(m, theta, tau_s, accept_prob) for m in range(len(p))],
        dtype=complex,
    )
    f_zero = np.asarray(
        [F_m(m, 0.0, tau_s, accept_prob).real for m in range(len(p))],
        dtype=float,
    )

    p_mon = float(np.dot(p, f_zero))
    if p_mon <= 0.0:
        raise ValueError("Monitor acceptance probability is zero.")

    return complex(np.dot(p, f_theta) / p_mon), p_mon


# ---------------------------------------------------------------------------
# Source-independent tail interface


def ideal_delta_exact(
    n_plus: int,
    tau_s: float,
    c_max: int,
) -> IdealTailResult:
    """
    Exact source-independent tail bound for the ideal upper-threshold monitor.

    Only for

        a_c = 1[c <= c_max],

    the global maximizer is

        m_star = c_max + n_plus + 1,

    and

        delta = Pr[Binomial(m_star,tau_s) > n_plus].
    """
    _check_tau(tau_s)
    _check_nonnegative_int("n_plus", n_plus)
    _check_nonnegative_int("c_max", c_max)

    m_star = c_max + n_plus + 1
    delta = float(binom.sf(n_plus, m_star, tau_s))

    return IdealTailResult(
        tau_s=float(tau_s),
        c_max=int(c_max),
        n_plus=int(n_plus),
        m_star=int(m_star),
        delta=delta,
    )


def conditional_tail_bound(delta: float, p_mon: float) -> float:
    """
    One-arm conditional tail:

        Pr(N>N_plus | A_mon) <= min(1, delta/P_mon).
    """
    _check_probability("delta", delta)
    _check_probability("p_mon", p_mon, positive=True)
    return min(1.0, float(delta) / float(p_mon))


def two_arm_union_tail_bound(
    delta_a: float,
    delta_b: float,
    p_mon_both: float,
) -> float:
    """
    Conservative two-arm interface:

        Pr(
            N_A>N_A_plus or N_B>N_B_plus
            | both monitors accept
        )
        <= min(1, (delta_A + delta_B)/P_mon_both).

    This remains valid for arbitrary A-B source correlations.
    """
    _check_probability("delta_a", delta_a)
    _check_probability("delta_b", delta_b)
    _check_probability("p_mon_both", p_mon_both, positive=True)

    return min(
        1.0,
        (float(delta_a) + float(delta_b)) / float(p_mon_both),
    )


def ideal_two_arm_interface(
    *,
    tau_s_a: float,
    tau_s_b: float,
    c_max_a: int,
    c_max_b: int,
    nA_plus: int,
    nB_plus: int,
    p_mon_both: float,
) -> IdealTwoArmInterface:
    """
    Derive epsilon_5 for the ideal hard-threshold benchmark.

    p_mon_both must be an observed/lower-bounded probability that both local
    monitors accept.  It is not generated by the worst-case source model.
    """
    da = ideal_delta_exact(nA_plus, tau_s_a, c_max_a).delta
    db = ideal_delta_exact(nB_plus, tau_s_b, c_max_b).delta
    eps5 = two_arm_union_tail_bound(da, db, p_mon_both)

    return IdealTwoArmInterface(
        delta_a=da,
        delta_b=db,
        epsilon_5=eps5,
        p_mon_both=float(p_mon_both),
        nA_plus=int(nA_plus),
        nB_plus=int(nB_plus),
    )


def scan_general_delta(
    n_plus: int,
    tau_s: float,
    accept_prob: Callable[[int], float],
    m_values: Iterable[int],
) -> tuple[int, float]:
    """
    Explore max_m H_m over a FINITE user-supplied set.

    IMPORTANT: this is only a diagnostic for a general response a_c.  Unless
    a separate proof bounds the supremum outside m_values, the result is a
    LOWER bound on sup_{m>=0} H_m, not a certified security upper bound.
    """
    _check_tau(tau_s)
    _check_nonnegative_int("n_plus", n_plus)

    best_m = None
    best_h = -1.0

    for m in m_values:
        _check_nonnegative_int("m", m)
        h = general_H_m(m, n_plus, tau_s, accept_prob)
        if h > best_h:
            best_h = h
            best_m = int(m)

    if best_m is None:
        raise ValueError("m_values must contain at least one value.")

    return best_m, float(best_h)


# ---------------------------------------------------------------------------
# Finite good-Fock support and overlap kernel


def good_fock_support(nA_plus: int, nB_plus: int) -> np.ndarray:
    """
    Rectangular good support

        0 <= n_A <= nA_plus,
        0 <= n_B <= nB_plus.

    Returns an array of shape (NQ,2).
    """
    _check_nonnegative_int("nA_plus", nA_plus)
    _check_nonnegative_int("nB_plus", nB_plus)

    return np.asarray(
        [
            (nA, nB)
            for nA in range(nA_plus + 1)
            for nB in range(nB_plus + 1)
        ],
        dtype=int,
    )


def good_overlap_kernel(
    phases: Sequence[tuple[float, float]],
    pairs: Sequence[tuple[int, int]],
    support: np.ndarray,
) -> np.ndarray:
    """
    Exact finite-Fock overlap kernel

        K[(s,t),(n_A,n_B)]
          = exp(i Delta phi_A n_A + i Delta phi_B n_B).
    """
    k = np.empty((len(pairs), len(support)), dtype=complex)

    nA = support[:, 0]
    nB = support[:, 1]

    for i, (s, t) in enumerate(pairs):
        dA = phases[t][0] - phases[s][0]
        dB = phases[t][1] - phases[s][1]
        k[i, :] = np.exp(1j * (dA * nA + dB * nB))

    return k


# ---------------------------------------------------------------------------
# Complete Stage-5A tail-relaxed Gram SDP


def solve_stage5a_sdp(
    *,
    M: int,
    nA_plus: int,
    nB_plus: int,
    epsilon_5: float,
    prob_of: dict[int, Sequence[float]],
    p_mon: float | None = None,
    f_ec: float = F_EC,
    solver=None,
    solver_opts=None,
) -> Stage5AResult:
    """
    Solve the complete finite Stage-5A Gram outer relaxation.

    Inputs
    ------
    M
        Number of phase bases.  There are NS=4*M signal labels.
    nA_plus, nB_plus
        Certified good-component surviving-photon cutoffs.
    epsilon_5
        Certified two-arm conditional bad mass:

            Pr(
                N_A>nA_plus or N_B>nB_plus
                | A_mon
            ) <= epsilon_5.

    prob_of
        Accepted-ensemble observed probabilities.  Keys must be
        Z_P, Z_M, Z_F and each value must have length 4*M:

            prob_of[z][s] = P(z | s, A_mon).

    p_mon
        Optional observed physical monitor acceptance probability.  If
        supplied, R_original = p_mon * R_acc is reported.  It does not enter
        the conditional Gram SDP itself.

    The SDP variables are
        G      : Gram matrix of central-node residual vectors,
        q      : unnormalised good accepted Fock masses,
        beta   : arbitrary bad accepted mass.

    Constraints are
        G >= 0,
        diag(G) = observed accepted-ensemble statistics,
        q >= 0,
        0 <= beta <= epsilon_5,
        sum(q) + beta = 1,
        |Gamma_st(G) - Lambda_good_st(q)| <= beta.

    The incoming source distribution p_{m_A,m_B} does not appear.  This is an
    intentional outer relaxation of the exact PNA-BS map.
    """
    if M < 1:
        raise ValueError("M must be positive.")
    _check_nonnegative_int("nA_plus", nA_plus)
    _check_nonnegative_int("nB_plus", nB_plus)
    _check_probability("epsilon_5", epsilon_5)

    if p_mon is not None:
        _check_probability("p_mon", p_mon, positive=True)

    ns = 4 * M
    phases = state_phases(M)

    # Validate observed probabilities.
    probs = {}
    for z in Z_ALL:
        if z not in prob_of:
            raise ValueError(f"prob_of is missing announcement key {z}.")
        arr = np.asarray(prob_of[z], dtype=float)
        if arr.shape != (ns,):
            raise ValueError(
                f"prob_of[{z}] must have shape ({ns},), got {arr.shape}."
            )
        if np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
            raise ValueError("Observed probabilities must lie in [0,1].")
        probs[z] = np.clip(arr, 0.0, 1.0)

    total = sum(probs[z] for z in Z_ALL)
    if not np.allclose(total, 1.0, atol=1e-9):
        raise ValueError(
            "For each signal label, accepted-ensemble announcement "
            "probabilities must sum to 1."
        )

    # The key-basis pass probability is an observed constant.
    p_pass_mon = 0.25 * sum(
        probs[Z_P][ab] + probs[Z_M][ab]
        for ab in range(4)
    )
    if p_pass_mon <= 0.0:
        raise ValueError("Observed P_pass|mon must be positive.")

    # Bit error from the same key-basis observed statistics.  For 00 and 11,
    # '+' is the nominal correct result; for 01 and 10, '-' is correct.
    error_prob = 0.25 * (
        probs[Z_M][0]
        + probs[Z_P][1]
        + probs[Z_P][2]
        + probs[Z_M][3]
    )
    e_bit = float(error_prob / p_pass_mon)

    # Remove zero-norm residual vectors.  PSD forces their whole row/column to
    # zero, and removing them improves numerical strict feasibility.
    active = [
        (z, s)
        for z in Z_ALL
        for s in range(ns)
        if probs[z][s] > 1e-12
    ]
    pos = {zs: k for k, zs in enumerate(active)}
    dim = len(active)

    G = cp.Variable((dim, dim), hermitian=True)
    cons = [G >> 0]

    def ent(z1: int, s1: int, z2: int, s2: int):
        k1 = pos.get((z1, s1))
        k2 = pos.get((z2, s2))
        if k1 is None or k2 is None:
            return 0
        return G[k1, k2]

    # Observed accepted-ensemble announcement probabilities.
    cons.append(
        cp.real(cp.diag(G))
        == np.asarray([probs[z][s] for (z, s) in active])
    )

    # Finite good accepted-state Fock masses plus arbitrary bad mass.
    support = good_fock_support(nA_plus, nB_plus)
    nq = len(support)

    q = cp.Variable(nq, nonneg=True)
    beta = cp.Variable(nonneg=True)

    cons += [
        cp.sum(q) + beta == 1.0,
        beta <= epsilon_5,
    ]

    # Every unordered signal pair gets one overlap-conservation SOC.
    pairs = [
        (s, t)
        for s in range(ns)
        for t in range(s + 1, ns)
    ]

    kernel = good_overlap_kernel(phases, pairs, support)
    good_overlap = kernel @ q

    gram_terms = []
    for s, t in pairs:
        gram_terms.append(sum(ent(z, s, z, t) for z in Z_ALL))
    gram_overlap = cp.hstack(gram_terms)

    diff = gram_overlap - good_overlap
    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(
        cp.norm(resid, 2, axis=1)
        <= beta + SLACK_FLOOR
    )

    # A7 objective on key-basis labels 00,01,10,11 = 0,1,2,3.
    bracket = (
        ent(Z_P, 0, Z_P, 3)
        - ent(Z_M, 0, Z_M, 3)
        - ent(Z_P, 1, Z_P, 2)
        + ent(Z_M, 1, Z_M, 2)
    )
    objective_scalar = cp.real(bracket)

    prob = cp.Problem(cp.Maximize(objective_scalar), cons)

    if solver is None:
        try:
            import mosek  # noqa: F401

            solver = cp.MOSEK
            opts = {}
        except ImportError:
            solver = cp.CLARABEL
            opts = {}
    else:
        opts = {}

    if solver_opts:
        opts = {**opts, **solver_opts}
    if solver != cp.MOSEK:
        opts.pop("mosek_params", None)

    prob.solve(solver=solver, **opts)

    base_diag = {
        "epsilon_5": float(epsilon_5),
        "beta": None,
        "good_mass": None,
        "n_good_states": int(nq),
        "gram_dim": int(dim),
        "n_pairs": int(len(pairs)),
        "p_mon": None if p_mon is None else float(p_mon),
        "objective_scalar": None,
    }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return Stage5AResult(
            e_ph_upper=1.0,
            e_bit=e_bit,
            p_pass_mon=float(p_pass_mon),
            r_acc=0.0,
            r_original=None if p_mon is None else 0.0,
            status=prob.status,
            q_good=None,
            support=support,
            diagnostics=base_diag,
        )

    # A7:
    #   e_ph = 1/2 + Re(bracket)/(4 P_pass|mon).
    e_ph_upper = float(
        np.clip(
            0.5 + float(prob.value) / (4.0 * p_pass_mon),
            0.0,
            1.0,
        )
    )

    # For privacy amplification, if the only information is e_ph <= u with
    # u >= 1/2, the worst binary entropy in that interval is h2(1/2)=1, so
    # the key rate is zero.  Do not feed h2(u>1/2) directly into the rate.
    if e_ph_upper >= 0.5:
        privacy_term = 0.0
    else:
        privacy_term = max(
            0.0,
            1.0 - h2(e_ph_upper) - f_ec * h2(e_bit),
        )

    r_acc = float(p_pass_mon * privacy_term)
    r_original = None if p_mon is None else float(p_mon * r_acc)

    q_val = np.asarray(q.value).ravel()
    diagnostics = {
        **base_diag,
        "beta": float(beta.value),
        "good_mass": float(np.sum(q_val)),
        "objective_scalar": float(prob.value),
    }

    return Stage5AResult(
        e_ph_upper=e_ph_upper,
        e_bit=e_bit,
        p_pass_mon=float(p_pass_mon),
        r_acc=r_acc,
        r_original=r_original,
        status=prob.status,
        q_good=q_val,
        support=support,
        diagnostics=diagnostics,
    )


def solve_stage5a_synthetic(
    *,
    M: int,
    n_bar: float,
    distance_km: float,
    nA_plus: int,
    nB_plus: int,
    epsilon_5: float,
    p_mon: float | None = None,
    solver=None,
    solver_opts=None,
) -> Stage5AResult:
    """
    Convenience wrapper using synthetic accepted-ensemble central statistics.

    The monitor security interface (nA_plus,nB_plus,epsilon_5) is supplied
    independently.  In particular, epsilon_5 is NOT inferred from n_bar.
    """
    prob_of, _, _ = synthetic_announcement_probabilities(
        M,
        n_bar,
        distance_km,
    )

    return solve_stage5a_sdp(
        M=M,
        nA_plus=nA_plus,
        nB_plus=nB_plus,
        epsilon_5=epsilon_5,
        prob_of=prob_of,
        p_mon=p_mon,
        solver=solver,
        solver_opts=solver_opts,
    )


def solve_stage5a_ideal_benchmark(
    *,
    M: int,
    n_bar: float,
    distance_km: float,
    tau_s_a: float,
    tau_s_b: float,
    c_max_a: int,
    c_max_b: int,
    nA_plus: int,
    nB_plus: int,
    p_mon_both: float,
    solver=None,
    solver_opts=None,
) -> tuple[Stage5AResult, IdealTwoArmInterface]:
    """
    Convenience wrapper for the IDEAL hard-threshold architecture benchmark.

    It first computes

        epsilon_5
          = min(1, [delta_A + delta_B] / P_mon,both)

    from the closed-form ideal deltas, then solves the tail-relaxed Gram SDP
    using synthetic accepted-ensemble central-node statistics.

    This is NOT yet the realistic Stage-5A detector calculation.
    """
    interface = ideal_two_arm_interface(
        tau_s_a=tau_s_a,
        tau_s_b=tau_s_b,
        c_max_a=c_max_a,
        c_max_b=c_max_b,
        nA_plus=nA_plus,
        nB_plus=nB_plus,
        p_mon_both=p_mon_both,
    )

    result = solve_stage5a_synthetic(
        M=M,
        n_bar=n_bar,
        distance_km=distance_km,
        nA_plus=nA_plus,
        nB_plus=nB_plus,
        epsilon_5=interface.epsilon_5,
        p_mon=p_mon_both,
        solver=solver,
        solver_opts=solver_opts,
    )

    return result, interface


# ---------------------------------------------------------------------------
# Explicit execution entry point


def main() -> None:
    """
    Example configuration.

    The example is intentionally kept behind the __main__ guard.  Importing
    this module performs no scan and no SDP solve.
    """
    M = M_DEFAULT
    n_bar = 0.01
    distance_km = 0.0

    # Ideal architecture benchmark only.
    tau_s = 1e-8
    c_max = 1_000_000
    n_plus = 5

    # Placeholder observed probability that BOTH local monitors accept.
    # Replace by an experimentally justified value before interpreting a
    # per-original-round rate.
    p_mon_both = 0.99

    interface = ideal_two_arm_interface(
        tau_s_a=tau_s,
        tau_s_b=tau_s,
        c_max_a=c_max,
        c_max_b=c_max,
        nA_plus=n_plus,
        nB_plus=n_plus,
        p_mon_both=p_mon_both,
    )

    print("Stage 5A ideal architecture benchmark")
    print(f"M             = {M}")
    print(f"tau_s         = {tau_s:g}")
    print(f"c_max         = {c_max}")
    print(f"N_plus        = {n_plus}")
    print(f"P_mon,both    = {p_mon_both:g}")
    print(f"delta_A       = {interface.delta_a:.6e}")
    print(f"delta_B       = {interface.delta_b:.6e}")
    print(f"epsilon_5     = {interface.epsilon_5:.6e}")
    print()
    print(
        "The full Stage-5A SDP is implemented in solve_stage5a_sdp(). "
        "This example does not call the solver automatically."
    )
    print(
        "To test the ideal benchmark explicitly, call "
        "solve_stage5a_ideal_benchmark(...) from a Python session."
    )


if __name__ == "__main__":
    main()
