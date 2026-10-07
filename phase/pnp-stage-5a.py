"""
Preliminary Stage-5A tools: joint trusted beam splitter + destructive PNA.

This file implements the exact one-arm beam-splitter/PNA formulas developed
in the working notes.  It is intentionally preliminary: it does NOT yet build
or solve the full Gram phase-error SDP, and it does NOT yet contain a realistic
detector model.

Stage 5A has no separate attenuator.  A single trusted beam splitter with
signal-port transmissivity tau_s sends the complementary fraction 1-tau_s to
the monitor.  The monitor acts before the secret phase encoding.

Notation
--------
M : incoming photon number
C : monitor-port photon number
N : surviving signal photon number

For an ideal lossless beam splitter M = C + N and

    N | M=m ~ Binomial(m, tau_s).

A general phase-insensitive monitor response is described by

    a_c = Pr(A_mon | C=c).

For an arbitrary incoming photon-number distribution p_m, the accepted signal
photon-number distribution and conditional phase overlaps can then be computed
exactly.

The first architecture sanity check is the ideal upper-threshold response

    a_c = 1[c <= c_max].

For this special case the source-independent tail

    delta(N_plus)
      = sup_m Pr(A_mon and N > N_plus | M=m)

has an exact maximizer

    m_star = c_max + N_plus + 1,

and therefore

    delta_ideal(N_plus)
      = Pr[Binomial(m_star, tau_s) > N_plus].

This is the quantity to inspect before adding detector inefficiency, noise,
finite resolution, saturation, etc.

IMPORTANT
---------
No numerical scan is run by importing this file.  The guarded main() below
only runs if the file is explicitly executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Basic data containers


@dataclass(frozen=True)
class IdealTailResult:
    tau_s: float
    c_max: int
    n_plus: int
    m_star: int
    delta: float


# ---------------------------------------------------------------------------
# Validation helpers


def _check_tau(tau_s: float) -> None:
    if not (0.0 < tau_s < 1.0):
        raise ValueError("tau_s must lie strictly between 0 and 1.")


def _check_nonnegative_int(name: str, value: int) -> None:
    if int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


# ---------------------------------------------------------------------------
# Monitor response models


def ideal_upper_response(c: int, c_max: int) -> float:
    """Ideal upper-threshold monitor: a_c = 1[c <= c_max]."""
    _check_nonnegative_int("c", c)
    _check_nonnegative_int("c_max", c_max)
    return 1.0 if c <= c_max else 0.0


# ---------------------------------------------------------------------------
# Exact fixed-m beam-splitter / monitor probabilities


def ideal_H_m(m: int, n_plus: int, tau_s: float, c_max: int) -> float:
    """
    Exact
        H_m(N_plus) = Pr(C <= c_max, N > N_plus | M=m)
    for the ideal upper-threshold monitor.

    Since N ~ Binomial(m, tau_s), the joint event is
        N >= max(N_plus + 1, m - c_max).
    """
    _check_tau(tau_s)
    for name, value in (("m", m), ("n_plus", n_plus), ("c_max", c_max)):
        _check_nonnegative_int(name, value)

    if m <= n_plus:
        return 0.0

    lower = max(n_plus + 1, m - c_max)
    if lower > m:
        return 0.0

    # sf(k) = Pr[N > k], so sf(lower - 1) = Pr[N >= lower].
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

    This routine is useful for detector models where a_c is not a sharp step.
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


# ---------------------------------------------------------------------------
# Source-independent ideal tail bound


def ideal_delta_exact(
    n_plus: int,
    tau_s: float,
    c_max: int,
) -> IdealTailResult:
    """
    Exact source-independent tail bound for the ideal upper-threshold monitor.

    The global maximizer over arbitrary incoming Fock number is

        m_star = c_max + n_plus + 1,

    and

        delta = Pr[Binomial(m_star, tau_s) > n_plus].

    Equivalently,
        delta = Pr[Binomial(m_star, 1-tau_s) <= c_max].
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
    Convert an unconditional accepted-bad probability bound into

        Pr(N > N_plus | A_mon) <= delta / P_mon.

    The result is clipped at 1 because probabilities cannot exceed 1.
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must lie in [0,1].")
    if not (0.0 < p_mon <= 1.0):
        raise ValueError("p_mon must lie in (0,1].")
    return min(1.0, float(delta) / float(p_mon))


def two_arm_union_tail_bound(
    delta_a: float,
    delta_b: float,
    p_mon_both: float,
) -> float:
    """
    Conservative two-arm union bound:

      Pr(N_A>N_A+ or N_B>N_B+ | both monitors accept)
        <= (delta_A + delta_B) / P_mon,both.
    """
    if not (0.0 <= delta_a <= 1.0 and 0.0 <= delta_b <= 1.0):
        raise ValueError("delta_a and delta_b must lie in [0,1].")
    if not (0.0 < p_mon_both <= 1.0):
        raise ValueError("p_mon_both must lie in (0,1].")
    return min(1.0, (float(delta_a) + float(delta_b)) / float(p_mon_both))


# ---------------------------------------------------------------------------
# Exact accepted distribution for a finite input truncation


def accepted_signal_distribution(
    p_m: Sequence[float],
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> tuple[np.ndarray, float]:
    """
    Compute the exact accepted surviving photon-number distribution for a
    finite input distribution p_m, m=0,...,M_max.

    Returns
    -------
    q_acc : ndarray
        Conditional distribution q_n^acc on n=0,...,M_max.
    p_mon : float
        Monitor acceptance probability.

    This is an exact calculation for the supplied finite p_m.  It is not a
    truncation certificate for an unknown infinite-tail source.
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

        q_tilde[: m + 1] += (
            pm
            * binom.pmf(n, m, tau_s)
            * a
        )

    p_mon = float(np.sum(q_tilde))
    if p_mon <= 0.0:
        raise ValueError("Monitor acceptance probability is zero.")

    return q_tilde / p_mon, p_mon


def accepted_overlap_from_q(q_acc: Sequence[float], theta: float) -> complex:
    """
    Exact common-purification phase overlap

        Lambda(theta) = sum_n q_n^acc exp(i theta n).
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


# ---------------------------------------------------------------------------
# Direct fixed-m conditional overlap kernel


def F_m(
    m: int,
    theta: float,
    tau_s: float,
    accept_prob: Callable[[int], float],
) -> complex:
    """
    Stage-5A fixed-input kernel

      F_m(theta)
        = sum_c a_c Bin(m,c;1-tau_s) exp[i theta (m-c)].

    F_m(0) is the acceptance probability A_m.
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
# General-response exploratory scan


def scan_general_delta(
    n_plus: int,
    tau_s: float,
    accept_prob: Callable[[int], float],
    m_values: Iterable[int],
) -> tuple[int, float]:
    """
    Explore max_m H_m over a user-supplied FINITE set of m values.

    IMPORTANT: for a general detector response this is only a finite scan.
    Unless a separate argument proves that the true maximizer lies inside
    m_values, the returned value is NOT a certified upper bound on
    sup_{m>=0} H_m.  It is a diagnostic/lower bound on that supremum.
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
# Explicit execution entry point
#
# Kept deliberately lightweight.  Nothing in this file has been run as part
# of the present commit.


def main() -> None:
    # Illustrative placeholders only.  Replace by a physically motivated
    # parameter choice before using the output in the security analysis.
    tau_s = 1e-8
    c_max = 1_000_000
    n_plus = 5

    result = ideal_delta_exact(
        n_plus=n_plus,
        tau_s=tau_s,
        c_max=c_max,
    )

    print("Stage 5A ideal upper-threshold sanity check")
    print(f"tau_s   = {result.tau_s:g}")
    print(f"c_max   = {result.c_max}")
    print(f"N_plus  = {result.n_plus}")
    print(f"m_star  = {result.m_star}")
    print(f"delta   = {result.delta:.6e}")
    print(
        "This is the exact ideal-monitor unconditional accepted-tail bound. "
        "A conditional bound additionally requires an observed P_mon."
    )


if __name__ == "__main__":
    main()
