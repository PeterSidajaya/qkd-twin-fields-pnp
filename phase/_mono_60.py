"""
Does modelling the attenuation at Alice's and Bob's side as a beamsplitter
change anything?

Claim under test: the SDP survives. Section V of arXiv:2012.11104 says the
channel-discrimination optimisation can no longer be cast as an SDP once
losses make the arriving state mixed, because Gram matrices need pure states
and the eigenbasis of the mixed state is not the Fock basis.

That obstruction does not apply here, because the loss channel is phase
covariant:

    L_t(exp(i theta n) rho exp(-i theta n))
        = exp(i theta n) L_t(rho) exp(-i theta n)

which follows from a^k exp(i theta n) = exp(-i k theta) exp(i theta n) a^k on
the Kraus operators, and holds with any spectator system along for the ride.

Setting it up correctly matters, and the obvious version is wrong. Do NOT
purify by keeping the reflected port: the beamsplitter B is unitary and
carries no dependence on the state index s, so

    <Phi_s1|Phi_s2> = <Psi|<0| U_s1^dag B^dag B U_s2 |Psi>|0> = Lambda(p)

and the attenuation buys nothing. That is correct physics for that
formulation - keeping both output ports means nothing was lost.

The right construction traces out the reflected modes first, which is what
actually happens, and purifies afterwards with a fictitious reference.

Eve holds a purification E0 of the light she sends, so the object is
rho^{A B E0}. Alice and Bob modulate, then attenuate, then the reflected
modes are discarded. Phase covariance survives the spectator E0, so

    rho_s^{T E0} = W_s rho_0^{T E0} W_s^dag,
    W_s = exp(i (phi^s_A n_TA + phi^s_B n_TB)) tensor 1_{E0}

with rho_0^{T E0} independent of s: every state is a phase rotation of ONE
mixed state, which is exactly the structure Section V lacks. Purify that with
a fictitious R', and |chi_s> = (W_s tensor 1_R') |chi> gives

    <chi_s1|chi_s2> = Tr[rho_0^{T E0} W_s1^dag W_s2]
                    = Tr_T[rho_0^T exp(i dphi . n)]
                    = sum_k q_k exp(i dphi k)

with q the TRANSMITTED Fock distribution. Different purifications of the same
rho^{A'B'TE0} agree on every marginal that matters, and e_ph is a property of
Alice and Bob's marginal, so nothing is lost by working with |chi_s>.

Eve's isometry acts on T and E0 with R' a spectator, so the isometry
constraint sum_z <e^z_s1|e^z_s2> = Lambda(q) holds with the Gram vectors
living in E_out tensor R'.

One caveat, worth being explicit about. Relaxing to any PSD Gram matrix on
E_out tensor R' hands Eve R', which is morally the lost light. That is
generous and therefore safe, but it means the gain here does NOT come from
Eve being denied the reflected photons. It comes entirely from q being
confined to the image of B_t, a genuine restriction on which Fock
distributions can reach the node. "Attenuation hides information from Eve" is
the intuitive story and it is not the one this proof uses.

So the only change is one linear map, their Eq. 43 in two modes:

    q_n = sum_{m >= n} p_m C(m,n) t^(2n) (1 - t^2)^(m-n)

Truncation needs care. Alice monitors what LEAVES her lab, so the output
energy is bounded by n_bar/2 per arm and the input by n_bar/(2 t^2), which
grows as t falls. The input therefore needs a much larger grid than the
output, and the energy relaxation must be applied on the input side, where
"outside the grid" does imply m_A + m_B >= N_IN + 1. Applying it to the
output would be unsound: the input tail produces output at every (a, b),
including (0, 0), so the missing output mass is not concentrated at high
photon number.
"""

import numpy as np
import cvxpy as cp
import math
from scipy.stats import binom

M = 4
N_OUT = 7                # diagnostics only now; not in any constraint
N_IN = 60
F_EC = 1.15
ETA_DET = 0.85
XI = 0.2
E_ALI = 0.015
P_DC = 5e-8
LOSS_ONLY = False

# Lower bound on the per-pair truncation slack; see the note where it is used.
SLACK_FLOOR = 1e-12

# Input grids used by the monotonicity check at the end of the run.
MONO_N_IN = (40, 60, 80)   # slow without MOSEK; trim if needed

Z_P, Z_M, Z_F = 0, 1, 2

out_pairs = [(a, b) for a in range(N_OUT + 1) for b in range(N_OUT + 1)
             if a + b <= N_OUT]
in_pairs = [(a, b) for a in range(N_IN + 1) for b in range(N_IN + 1)
            if a + b <= N_IN]
NQ, NP = len(out_pairs), len(in_pairs)
QA = np.array([o[0] for o in out_pairs])
QB = np.array([o[1] for o in out_pairs])
PA = np.array([i[0] for i in in_pairs])
PB = np.array([i[1] for i in in_pairs])

enc_phases = [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]


_ATT_CACHE = {}


def attenuation_matrix(t2):
    """
    B[n_out, m_in] for the two-mode loss channel, their Eq. 43.

    The channel factorises across modes, so build the single-mode matrix once
    and take the product of the two factors. Doing it pair by pair with
    scipy.stats.binom.pmf is the dominant cost of the whole script at large
    N_IN.
    """
    if t2 in _ATT_CACHE:
        return _ATT_CACHE[t2]
    n = np.arange(N_OUT + 1)[:, None]
    m = np.arange(N_IN + 1)[None, :]
    single = binom.pmf(n, m, t2)                  # (N_OUT+1, N_IN+1)
    single = np.nan_to_num(single)
    B = single[QA][:, PA] * single[QB][:, PB]
    _ATT_CACHE[t2] = B
    return B


def state_phases(M):
    out = []
    for m in range(M):
        phi = np.pi * m / M
        for dA, dB in enc_phases:
            out.append((phi + dA, phi + dB))
    return out


def h2(e):
    return 0.0 if e <= 0 or e >= 1 else -e * math.log2(e) - (1 - e) * math.log2(1 - e)


def solve(M, n_bar, L, t2):
    phases = state_phases(M)
    NS = 4 * M
    eta = ETA_DET * 10 ** (-XI * L / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI
    P_sig = 1.0 - np.exp(-eta * n_bar)
    P_pass = P_sig + p_dc
    e_obs = (e_ali * P_sig + 0.5 * p_dc) / P_pass

    P_plus = np.zeros(NS); P_minus = np.zeros(NS)
    for s in range(NS):
        hit, miss = (1 - e_obs) * P_pass, e_obs * P_pass
        P_plus[s], P_minus[s] = (hit, miss) if (s % 4) in (0, 3) else (miss, hit)
    P_fail = 1.0 - P_plus - P_minus
    prob_of = {Z_P: P_plus, Z_M: P_minus, Z_F: P_fail}

    active = [(z, s) for z in (Z_P, Z_M, Z_F) for s in range(NS)
              if prob_of[z][s] > 1e-12]
    pos = {zs: k for k, zs in enumerate(active)}
    DIM = len(active)

    G = cp.Variable((DIM, DIM), hermitian=True)
    cons = [G >> 0]

    def ent(z1, s1, z2, s2):
        k1, k2 = pos.get((z1, s1)), pos.get((z2, s2))
        return 0 if (k1 is None or k2 is None) else G[k1, k2]

    p = cp.Variable(NP, nonneg=True)
    cons.append(cp.sum(p) <= 1)
    eps_in = 1 - cp.sum(p)

    B = attenuation_matrix(t2)     # diagnostics only; not in any constraint


    # Energy. Input relaxation is the sound one: outside the input grid,
    # m_A + m_B >= N_IN + 1. Per-arm constraints drop the tail rather than
    # charging it, on both sides.
    # Alice monitors what leaves her lab, and E_q[n_A] = t^2 E_p[m_A]
    # exactly, so the output constraint is the input constraint scaled. No
    # need to form q at all here; it is built below for diagnostics only.
    cons.append(PA @ p <= n_bar / (2 * t2))
    cons.append(PB @ p <= n_bar / (2 * t2))
    cons.append((PA + PB) @ p + (N_IN + 1) * eps_in <= n_bar / t2)

    # Overlaps in closed form, with no output truncation at all.
    #
    # An input Fock state m contributes, in one mode,
    #   sum_n Bin(n | m, t^2) exp(i dphi n) = (1 - t^2 + t^2 exp(i dphi))^m
    # exactly. Summing the binomial numerically and absorbing the remainder
    # into slack throws that away: the leftover was then bounded by its raw
    # mass, which is far looser and, worse, destroys nesting in N_IN, since
    # the difference between two truncated sums carries no decay factor. That
    # is what made e_ph non-monotone (0.2305, 0.2104, 0.2144 at N_IN = 40,
    # 60, 80) even with every solve reporting optimal.
    #
    # Written this way the only slack left is the input tail, N_OUT drops out
    # of the constraints entirely, and the feasible sets are properly nested.
    pairs = [(s1, s2) for s1 in range(NS) for s2 in range(s1 + 1, NS)]
    C = np.zeros((len(pairs), NP), dtype=complex)
    zmax = np.zeros(len(pairs))
    for i, (s1, s2) in enumerate(pairs):
        dA = phases[s2][0] - phases[s1][0]
        dB = phases[s2][1] - phases[s1][1]
        zA = 1 - t2 + t2 * np.exp(1j * dA)
        zB = 1 - t2 + t2 * np.exp(1j * dB)
        C[i, :] = zA ** PA * zB ** PB
        zmax[i] = max(abs(zA), abs(zB))

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
    # Per-pair truncation slack.
    #
    # An input Fock state m contributes to the overlap, in one mode,
    #   sum_n Bin(n | m, t^2) exp(i dphi n) = (1 - t^2 + t^2 exp(i dphi))^m
    # so its modulus decays as |z|^m with |z| = |1 - t^2 + t^2 e^{i dphi}|,
    # which is strictly below 1 for any dphi != 0 once t^2 < 1. Outside the
    # input grid m_A + m_B >= N_IN + 1, and putting every photon in whichever
    # mode has the larger |z| is the worst case, so the input tail can
    # contribute at most eps_in * max(|z_A|, |z_B|)^(N_IN+1).
    #
    # This is the decay factor of Eq. 9 of arXiv:2012.11104, which pure phase
    # modulation destroys: there the per-sector overlap has |k| = 1 exactly,
    # so no decay is available and the slack is the bare tail mass.
    # Attenuation brings it back, because |z| < 1.
    #
    # The other missing piece is output that falls outside the output grid
    # from inputs that are inside the input grid. That mass is sum(p) -
    # sum(q), linear in p, and is bounded by its own modulus.
    decay = zmax ** (N_IN + 1)

    # Clamp the slack from below. The decay factors span many orders of
    # magnitude across pairs: at t^2 = 0.1 a phase difference of pi gives
    # |z| = 0.8 while pi/4 gives 0.973, and at N_IN = 100 those are 1e-10 and
    # 0.065. Multiplied by an eps_in heading to zero, the smallest cone radii
    # fall below any solver's feasibility tolerance while the Gram entries
    # are order 1e-2. Such a cone is numerically an equality already, so
    # saying so explicitly is both honest and better conditioned than leaving
    # it to the solver to discover. Check that the answer does not move when
    # SLACK_FLOOR is varied; if it does, the floor is doing real work and is
    # too large.
    # Added, not maximised: cp.maximum of an affine expression and a constant
    # is convex, and a convex right-hand side is not DCP. Both slack terms are
    # non-negative, so adding the floor differs from taking the maximum by at
    # most SLACK_FLOOR itself, and it enlarges the slack, which is the safe
    # direction.
    slack = decay * eps_in + SLACK_FLOOR

    diff = sum(terms) - C @ p
    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(cp.norm(resid, 2, axis=1) <= slack)

    cons.append(cp.real(cp.diag(G)) ==
                np.array([prob_of[z][s] for (z, s) in active]))

    bracket = (ent(Z_P, 0, Z_P, 3) - ent(Z_M, 0, Z_M, 3)
               - ent(Z_P, 1, Z_P, 2) + ent(Z_M, 1, Z_M, 2))
    obj = cp.real(bracket)
    cons.append(obj <= 0.0)

    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        import mosek
        prob.solve(solver=cp.MOSEK)
    except ImportError:
        prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return 0.5, e_obs, P_pass, prob.status, None, None
    e_ph = float(np.clip(0.5 + prob.value / (4 * P_pass), 0.0, 0.5))
    return e_ph, e_obs, P_pass, prob.status, p.value, np.asarray(B @ p.value)


def eps_in_bound(n_bar, t2, N_IN):
    """
    Worst case the energy relaxation permits for the input tail:
    eps_in <= n_bar / (t^2 (N_IN + 1)).

    This is a Markov-type bound and the solver usually lands orders of
    magnitude below it, so it is a ceiling to report rather than a target to
    design around. Choose N_IN by scanning upward until e_ph stops moving,
    then stop: going further shrinks the decay factors, drives the smallest
    cone radii to zero and makes the solver return eps_in < 0, meaning
    sum(p) > 1.
    """
    return n_bar / (t2 * (N_IN + 1))


