"""
Plug-and-play phase-matching QKD with an untrusted source: phase-error SDP.

Structure follows the Fig. 6 reproduction, with the one change that matters:
the source is Eve's, so the pairwise overlaps of the sent states are not
known constants but linear functions of an optimisation variable p, and the
inner-product constraints become second-order cones with slack epsilon
instead of equalities.

Protocol. The central node supplies the light. Alice and Bob each apply a
phase and return their pulse; the node announces Psi+, Psi- or fail.

    basis m in {0, ..., M-1},  phase  phi_m = m pi / M
    bit   b in {0, 1},         extra shift  b pi

Basis 0 generates key. Bases 1..M-1 are test bases: their rounds are
sacrificed to bound the phase error, exactly as in the paper. This is the
change from the earlier plug-and-play code, where all M bases carried key
and the rate paid a 1/M sifting factor. With the basis choice biased towards
basis 0, the key-round fraction tends to 1 asymptotically, so no sifting
factor appears in the rate below.

Matched bases only (x = y). In the reproduction this cost nothing:
R/PLOB came out 0.083, 0.114, 0.247 at 10, 20, 30 dB against 0.083, 0.115,
0.247 with all basis combinations, at a quarter of the problem size.

State set: 4M joint states indexed by s = 4m + enc, with enc enumerating
(b_A, b_B) as (0,0), (0,1), (1,0), (1,1).

Why an untrusted source is still tractable. The modulation exp(i phi n) is
diagonal in Fock, and the same state sits on both sides of every overlap, so

    <psi_s1|psi_s2> = sum_k p_k exp(i (dphi_A n_A^k + dphi_B n_B^k))

with p_k the Fock distribution of whatever arrives. Nothing about the source
beyond p enters, and p is bounded only by what Alice and Bob monitor, namely
the mean energy leaving each lab.

Two things carried over from the reproduction because they are structural,
not incidental to that protocol:

  - zero-norm reduction. A vector |e^z_s> whose probability is exactly zero
    puts a zero on the Gram diagonal, which by positive semidefiniteness
    forces its whole row to vanish. The feasible set then has empty relative
    interior, Slater's condition fails, and interior-point solvers either
    error out or return a spurious e_ph = 1/2. Dropping those indices is
    exact. This bites whenever LOSS_ONLY is set.

  - the inner-product constraints are built with sparse elementwise indexing
    G[rows, cols], never as a matrix product against the Hermitian variable.
    The latter canonicalises into dense coefficient blocks spanning all
    dim(G)^2 variables and reaches tens of GB by M = 3.
"""

import numpy as np
import cvxpy as cp
import math

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

M = 5                  # bases; basis 0 is key, 1..M-1 are test
N_MAX = 7              # Fock truncation, n_A + n_B <= N_MAX
F_EC = 1.15

# Detector and channel, as in the earlier plug-and-play code.
ETA_DET = 0.85
XI = 0.2               # dB/km
E_ALI = 0.015
P_DC = 5e-8

# Noiseless mode. Worth running first: it isolates the cost of the untrusted
# source from the cost of detector imperfections, and it is the case where
# the zero-norm reduction matters.
LOSS_ONLY = False

# "eve"   - p is an optimisation variable. This is the real problem.
# "fixed" - p is a coherent source of the same mean energy, so Lambda is a
#           constant and epsilon is the genuine Poisson tail. This is the
#           trusted-source version of the same protocol. Run it first: it is
#           the reference the untrusted case is measured against, and the gap
#           between them is exactly what the untrusted source costs.
SOURCE = "eve"

Z_P, Z_M, Z_F = 0, 1, 2

fock_pairs = [(a, b) for a in range(N_MAX + 1) for b in range(N_MAX + 1)
              if a + b <= N_MAX]
NF = len(fock_pairs)
NA_K = np.array([f[0] for f in fock_pairs])
NB_K = np.array([f[1] for f in fock_pairs])

enc_phases = [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]


def state_phases(M):
    """s = 4m + enc  ->  (phi_A, phi_B)."""
    out = []
    for m in range(M):
        phi_m = np.pi * m / M
        for dA, dB in enc_phases:
            out.append((phi_m + dA, phi_m + dB))
    return out


def h2(e):
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return -e * math.log2(e) - (1 - e) * math.log2(1 - e)


def channel(n_bar, L):
    """Honest channel: total click rate and error rate."""
    eta = ETA_DET * 10 ** (-XI * L / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI
    P_sig = 1.0 - np.exp(-eta * n_bar)
    P_pass = P_sig + p_dc
    e_obs = (e_ali * P_sig + 0.5 * p_dc) / P_pass
    return P_pass, e_obs


def coherent_p(n_bar):
    """Fock weights of a coherent source with n_bar/2 mean photons per arm."""
    mu = n_bar / 2.0
    w = np.array([
        np.exp(-mu) * mu ** a / math.factorial(a) *
        np.exp(-mu) * mu ** b / math.factorial(b)
        for a, b in fock_pairs
    ])
    return w


# --------------------------------------------------------------------------
# SDP
# --------------------------------------------------------------------------

def solve_eph(M, n_bar, L, solver=None, solver_opts=None):
    phases = state_phases(M)
    NS = 4 * M
    P_pass, e_obs = channel(n_bar, L)

    # Observed per-state detector statistics.
    #
    # enc 0 and 3 have phi_A - phi_B = 0, so Psi+ is the expected outcome and
    # Psi- is an error; enc 1 and 2 are the reverse. Imposing both diagonals
    # per state is the full distribution Alice and Bob see, and is tighter
    # than constraining only the total click rate and an averaged QBER.
    P_plus = np.zeros(NS)
    P_minus = np.zeros(NS)
    for s in range(NS):
        expects_plus = (s % 4) in (0, 3)
        hit, miss = (1.0 - e_obs) * P_pass, e_obs * P_pass
        P_plus[s], P_minus[s] = (hit, miss) if expects_plus else (miss, hit)
    P_fail = 1.0 - P_plus - P_minus
    prob_of = {Z_P: P_plus, Z_M: P_minus, Z_F: P_fail}

    # Zero-norm reduction (see module docstring).
    TOL = 1e-12
    active = [(z, s) for z in (Z_P, Z_M, Z_F) for s in range(NS)
              if prob_of[z][s] > TOL]
    pos = {zs: k for k, zs in enumerate(active)}
    DIM = len(active)

    G = cp.Variable((DIM, DIM), hermitian=True)
    cons = [G >> 0]

    def ent(z1, s1, z2, s2):
        k1, k2 = pos.get((z1, s1)), pos.get((z2, s2))
        if k1 is None or k2 is None:
            return 0
        return G[k1, k2]

    # --- the source ---
    if SOURCE == "eve":
        p = cp.Variable(NF, nonneg=True)
        cons.append(cp.sum(p) <= 1)
        eps = 1 - cp.sum(p)

        # Energy. S = {(n_A,n_B) : n_A+n_B <= N_MAX}; outside it only
        # n_A + n_B >= N_MAX+1 is known, so (N_MAX+1) eps may be charged to
        # the sum but not to either arm alone, since (0, N_MAX+1) lies
        # outside S with n_A = 0. Charging it per arm excludes physical
        # sources and inflates the rate.
        cons.append(NA_K @ p <= n_bar / 2)
        cons.append(NB_K @ p <= n_bar / 2)
        cons.append((NA_K + NB_K) @ p + (N_MAX + 1) * eps <= n_bar)
    else:
        p = coherent_p(n_bar)
        eps = float(max(0.0, 1.0 - p.sum()))

    # --- inner products ---
    #
    # Lambda_{s1,s2}(p) = sum_k p_k exp(i (dphi_A n_A^k + dphi_B n_B^k)),
    # linear in p. The isometry gives sum_z <e^z_s1|e^z_s2> = <psi_s1|psi_s2>,
    # and the truncated Lambda differs from the true overlap by at most eps,
    # so the constraint is |sum_z G - Lambda| <= eps, a second-order cone.
    #
    # Diagonal: the physical norm is exactly 1, so that one is an equality,
    # not relaxed. Truncation is an artefact of how p is parametrised.
    pairs = [(s1, s2) for s1 in range(NS) for s2 in range(s1 + 1, NS)]

    C = np.zeros((len(pairs), NF), dtype=complex)
    for i, (s1, s2) in enumerate(pairs):
        dA = phases[s2][0] - phases[s1][0]
        dB = phases[s2][1] - phases[s1][1]
        C[i, :] = np.exp(1j * (dA * NA_K + dB * NB_K))

    terms = []
    for z in (Z_P, Z_M, Z_F):
        rows, cols, mask = [], [], []
        for (s1, s2) in pairs:
            k1, k2 = pos.get((z, s1)), pos.get((z, s2))
            if k1 is None or k2 is None:
                rows.append(0); cols.append(0); mask.append(0.0)
            else:
                rows.append(k1); cols.append(k2); mask.append(1.0)
        terms.append(cp.multiply(np.array(mask), G[rows, cols]))
    lhs = sum(terms)

    if SOURCE == "eve":
        diff = lhs - C @ p
    else:
        diff = lhs - (C @ p)

    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T   # (len(pairs), 2)
    cons.append(cp.norm(resid, 2, axis=1) <= eps * np.ones(len(pairs)))

    diag_rows = [pos[(z, s)] for (z, s) in active]
    cons.append(cp.real(cp.diag(G)) ==
                np.array([prob_of[z][s] for (z, s) in active]))

    # Exact normalisation of the sent states: sum_z ||e^z_s||^2 = 1.
    # Implied by the diagonal constraints above plus P_plus + P_minus +
    # P_fail = 1, so it is not imposed separately.

    # --- objective: Eq. A7 on the key basis (m = 0) ---
    #
    #   e_ph = 1/2 + (1 / 4 P_pass) Re[ <e^+_{++}|e^+_{--}>
    #                                 - <e^-_{++}|e^-_{--}>
    #                                 - <e^+_{+-}|e^+_{-+}>
    #                                 + <e^-_{+-}|e^-_{-+}> ]
    #
    # Optimise the bracket, not e_ph: the two differ by an affine map whose
    # 1/(4 P_pass) factor reaches 250 by 30 dB and grows as 1/sqrt(eta),
    # which hands the solver coefficients that blow up exactly where the
    # problem is already badly scaled. The cap e_ph <= 1/2 becomes
    # Re[bracket] <= 0.
    #
    # NOT YET VERIFIED: A7 is derived in the paper's Appendix A for coherent
    # states |+-sqrt(mu)>. Here the key states are |psi> and exp(i pi n)|psi>
    # with psi carrying an unknown Fock distribution. The derivation is
    # written in terms of Eve's post-measurement states and the two-state
    # encoding, so it should carry over, but the whole objective rests on it
    # and it has not been redone.
    s_pp, s_pm, s_mp, s_mm = 0, 1, 2, 3          # basis 0, enc 0..3

    bracket = (ent(Z_P, s_pp, Z_P, s_mm)
               - ent(Z_M, s_pp, Z_M, s_mm)
               - ent(Z_P, s_pm, Z_P, s_mp)
               + ent(Z_M, s_pm, Z_M, s_mp))
    obj = cp.real(bracket)
    cons.append(obj <= 0.0)

    prob = cp.Problem(cp.Maximize(obj), cons)

    if solver is None:
        try:
            import mosek
            solver, opts = cp.MOSEK, {}
        except ImportError:
            solver, opts = cp.CLARABEL, {}
    else:
        opts = {}
    if solver_opts:
        opts = {**opts, **solver_opts}
    if solver is not cp.MOSEK:
        opts.pop("mosek_params", None)

    prob.solve(solver=solver, **opts)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return 0.5, e_obs, P_pass, prob.status, None
    e_ph = 0.5 + prob.value / (4.0 * P_pass)
    p_val = p.value if SOURCE == "eve" else p
    return (float(np.clip(e_ph, 0.0, 0.5)), e_obs, P_pass, prob.status, p_val)


def key_rate(M, n_bar, L):
    e_ph, e_bit, P_pass, status, p_val = solve_eph(M, n_bar, L)
    # No 1/M sifting factor: basis choice is biased towards basis 0, so the
    # key-round fraction tends to 1 asymptotically.
    R = P_pass * max(0.0, 1.0 - h2(e_ph) - F_EC * h2(e_bit))
    return R, e_ph, e_bit, P_pass, status, p_val


# --------------------------------------------------------------------------

if __name__ == "__main__":
    n_bar = 0.01

    print(f"M = {M} ({2*(M-1)} test states)   N_MAX = {N_MAX}   "
          f"n_bar = {n_bar}   source = {SOURCE}   "
          f"{'loss only' if LOSS_ONLY else 'with noise'}")
    print(f"{'L(km)':>7} {'e_ph':>8} {'e_bit':>8} {'P_pass':>11} "
          f"{'R':>11} {'eps':>10}  status")

    for L in (0.0, 2.0, 5.0, 10.0, 20.0):
        R, e_ph, e_bit, P_pass, status, p_val = key_rate(M, n_bar, L)
        eps = 1.0 - float(np.sum(p_val)) if p_val is not None else float("nan")
        print(f"{L:7.1f} {e_ph:8.4f} {e_bit:8.4f} {P_pass:11.3e} "
              f"{R:11.3e} {eps:10.3e}  {status}", flush=True)