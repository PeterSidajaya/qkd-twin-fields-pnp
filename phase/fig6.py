"""
Reproduction of Fig. 6 of Primaatmaja, Lavie, Goh, Wang, Lim,
"Versatile security analysis of measurement-device-independent quantum key
distribution" (arXiv:1901.01942v4): phase-matching MDI-QKD with a finite
number of test states, against the PLOB bound.

This is the TRUSTED-source case. Nothing here is plug-and-play. The point is
to check the A7 objective, the Gram indexing and the sign conventions against
a published answer before any of it is reused with an untrusted source.

Protocol (their Eqs. 9 and 13). Alice and Bob independently pick a basis
x, y in {0,...,M-1} and a bit a, b, and send

    |a> = |(-1)^a e^{i theta_x} sqrt(mu_x)>,   theta_x = x pi / M
    |b> = |(-1)^b e^{i theta_y} sqrt(mu_y)>,   theta_y = y pi / M

Basis 0 is the key basis; bases 1..M-1 are test bases, giving 2(M-1) test
states per party. So M = 2, 3, 4 are the "two / four / six test states"
curves of Fig. 6.

The central node interferes the two pulses on a 50:50 beamsplitter and
announces z in {Psi+, Psi-, fail}. Bob flips his bit on Psi-.

Why this is much smaller than the plug-and-play SDP: the source is trusted,
so the overlaps <psi_s1|psi_s2> are known coherent-state inner products in
closed form. There is no Fock distribution variable, no truncation, no
epsilon, and the inner-product constraints are equalities rather than second
order cones.

Channel: the node is equidistant, so each arm has transmittance sqrt(eta)
where eta is the total Alice-to-Bob transmittance. That convention is fixed
by their Eq. 15, whose prefactor 1 - exp(-2 mu sqrt(eta)) is the click
probability when mu sqrt(eta) arrives from each arm. Following Section IV C,
eta includes the detector efficiency and the plots run against total loss in
dB.
"""

import numpy as np
import cvxpy as cp
import math
import time
import itertools

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

# Bases are set per run in the __main__ block below (M_LIST): 2(M-1)
# test states, so M = 2, 3, 4 give the Fig. 6 curves.
F_EC = 1.0            # Shannon limit, as the paper assumes

# Fig. 6(a): loss only. Fig. 6(b): Parameter 2 of their Table I.
LOSS_ONLY = True

# Use only matched-basis rounds (x = y) instead of all (x, y) combinations.
# States drop from 4 M^2 to 4 M, so at M = 4 dim G falls from 176 to about
# 40 after the zero-norm reduction. That matters a lot: the solver's memory
# is driven by the real PSD block, which is 2 dim G, and 352 x 352 is where
# the run reaches tens of GB.
#
# Dropping the mismatched rounds removes constraints, so the bound can only
# get looser and the rate lower. It is therefore safe, and the question is
# whether it is still tight enough to reproduce the figure. Check at M = 2
# against the all-pairs numbers (R/PLOB 0.083, 0.115, 0.247, 0.520 at 10,
# 20, 30, 40 dB) before trusting M = 3 and 4.
MATCHED_BASES_ONLY = True
P_DC  = 0.0 if LOSS_ONLY else 5e-8
E_ALI = 0.0 if LOSS_ONLY else 0.015

# --------------------------------------------------------------------------
# State set
# --------------------------------------------------------------------------
# Alice's single-party states are indexed by (a, x); likewise Bob's (b, y).
# The joint state sent in a round is the product, so there are (2M)^2 = 4M^2
# joint states. Key rounds are x = y = 0.

def party_states(M, mus):
    """(a, x) -> complex amplitude alpha, in order a-major within each x."""
    out = []
    for x in range(M):
        theta = np.pi * x / M
        for a in (0, 1):
            out.append(((-1) ** a) * np.exp(1j * theta) * np.sqrt(mus[x]))
    return out

Z_P, Z_M, Z_F = 0, 1, 2          # Psi+, Psi-, fail
N_ANN = 3


def coherent_overlap(alpha, beta):
    """<alpha|beta> for single-mode coherent states."""
    return np.exp(-0.5 * abs(alpha) ** 2 - 0.5 * abs(beta) ** 2
                  + np.conj(alpha) * beta)


def detection_probs(aA, aB, eta, p_dc, e_ali):
    """
    P(Psi+), P(Psi-) for amplitudes aA, aB at the source.

    Each arm carries transmittance sqrt(eta), so the amplitude at the
    beamsplitter is scaled by eta**0.25. The beamsplitter outputs are
    (aA +/- aB)/sqrt(2) and each detector is a threshold detector.
    Conclusive means exactly one of the two fires.
    """
    s = eta ** 0.25
    g_p = (aA * s + aB * s) / np.sqrt(2.0)
    g_m = (aA * s - aB * s) / np.sqrt(2.0)
    n_p, n_m = abs(g_p) ** 2, abs(g_m) ** 2

    if e_ali > 0.0:
        # Misalignment as imperfect interference: a fraction e_ali of the
        # intensity lands in the wrong output port. The paper does not state
        # its misalignment model, so this is a choice, not a reproduction.
        n_p, n_m = ((1 - e_ali) * n_p + e_ali * n_m,
                    (1 - e_ali) * n_m + e_ali * n_p)

    d_p = 1.0 - (1.0 - p_dc) * np.exp(-n_p)      # P(detector + fires)
    d_m = 1.0 - (1.0 - p_dc) * np.exp(-n_m)
    return d_p * (1.0 - d_m), d_m * (1.0 - d_p)


def h2(e):
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return -e * math.log2(e) - (1 - e) * math.log2(1 - e)


# --------------------------------------------------------------------------
# SDP
# --------------------------------------------------------------------------

def solve_eph(M, mus, eta, p_dc=0.0, e_ali=0.0, solver=None, verbose=False,
              solver_opts=None):
    """Maximise the key-basis phase-error rate. Returns (e_ph, e_bit, P_pass)."""
    A = party_states(M, mus)
    B = party_states(M, mus)
    joint = list(itertools.product(range(len(A)), range(len(B))))
    if MATCHED_BASES_ONLY:
        # Keep only rounds where Alice and Bob chose the same basis, x = y.
        # Party index is 2*x + a, so x = i // 2.
        joint = [(i, j) for (i, j) in joint if i // 2 == j // 2]
    NS = len(joint)                       # 4 M^2


    # --- observed statistics, from the honest channel ---
    P_plus = np.zeros(NS)
    P_minus = np.zeros(NS)
    for s, (i, j) in enumerate(joint):
        P_plus[s], P_minus[s] = detection_probs(A[i], B[j], eta, p_dc, e_ali)
    P_fail = 1.0 - P_plus - P_minus
    prob_of = {Z_P: P_plus, Z_M: P_minus, Z_F: P_fail}

    # --- exact overlaps of the sent states (trusted source) ---
    ovl = np.zeros((NS, NS), dtype=complex)
    for s1, (i1, j1) in enumerate(joint):
        for s2, (i2, j2) in enumerate(joint):
            ovl[s1, s2] = (coherent_overlap(A[i1], A[i2])
                           * coherent_overlap(B[j1], B[j2]))

    # Keep only the vectors with nonzero norm.
    #
    # In the loss-only case many detection probabilities are exactly zero: a
    # key state with a = b interferes fully into the + port, so P(Psi-) = 0.
    # The corresponding |e^z_s> is the zero vector. Leaving it in G puts a
    # zero on the diagonal, which by positive semidefiniteness forces its
    # whole row and column to vanish, so the feasible set has empty relative
    # interior, Slater's condition fails and interior-point solvers error out.
    # Dropping those indices is exact, not an approximation: the entries they
    # would occupy are zero by construction and are treated as such below.
    TOL = 1e-12
    active = [(z, s) for z in (Z_P, Z_M, Z_F) for s in range(NS)
              if prob_of[z][s] > TOL]
    pos = {zs: k for k, zs in enumerate(active)}
    DIM = len(active)

    G = cp.Variable((DIM, DIM), hermitian=True)
    cons = [G >> 0]

    def ent(z1, s1, z2, s2):
        """Gram entry, or the constant 0 for a dropped (zero-norm) vector."""
        k1, k2 = pos.get((z1, s1)), pos.get((z2, s2))
        if k1 is None or k2 is None:
            return 0
        return G[k1, k2]

    # Isometry preserves inner products:
    #   sum_z <e^z_s1|e^z_s2> = <psi_s1|psi_s2>
    # Exact, because the source is trusted.
    #
    # Built as ONE vector equality over the 4M^2(4M^2+1)/2 upper-triangular
    # pairs, via elementwise indexing G[rows, cols]. Each row of the
    # resulting constraint touches at most three entries of G.
    #
    # Do NOT write this as sum_z S_z G S_z^T == ovl with selection matrices.
    # It says the same thing, but cvxpy canonicalises a matrix product
    # against a Hermitian variable into dense coefficient blocks spanning all
    # dim(G)^2 variables, which reaches tens of GB by M = 3. The sparsity
    # here is the whole point.
    pairs = [(s1, s2) for s1 in range(NS) for s2 in range(s1, NS)]
    ovl_vec = np.array([ovl[s1, s2] for (s1, s2) in pairs])

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
    cons.append(sum(terms) == ovl_vec)

    # Observed statistics, imposed per joint state. This is the full
    # distribution Alice and Bob see; the paper writes it coarse-grained as
    # P_pass^gamma and e_bit^gamma per basis pair.
    diag_target = np.array([prob_of[z][s] for (z, s) in active])
    cons.append(cp.real(cp.diag(G)) == diag_target)

    # --- key basis: x = y = 0 ---
    def key_state(a, b):
        ia = 2 * 0 + a          # x = 0
        ib = 2 * 0 + b          # y = 0
        return joint.index((ia, ib))

    s_pp = key_state(0, 0)      # a=0, b=0   ("++")
    s_pm = key_state(0, 1)      # a=0, b=1   ("+-")
    s_mp = key_state(1, 0)      # a=1, b=0   ("-+")
    s_mm = key_state(1, 1)      # a=1, b=1   ("--")

    P_pass_key = 0.25 * sum(P_plus[s] + P_minus[s]
                            for s in (s_pp, s_pm, s_mp, s_mm))

    # Bob flips on Psi-, so an error is Psi+ with a != b, or Psi- with a == b.
    err = (P_plus[s_pm] + P_plus[s_mp] + P_minus[s_pp] + P_minus[s_mm])
    e_bit = 0.25 * err / P_pass_key

    # Eq. A7:
    #   e_ph = 1/2 + (1 / 4 P_pass) Re[ <e^+_{++}|e^+_{--}>
    #                                 - <e^-_{++}|e^-_{--}>
    #                                 - <e^+_{+-}|e^+_{-+}>
    #                                 + <e^-_{+-}|e^-_{-+}> ]
    #
    # The plus sign is correct. Noiseless check: ++ and -- both announce
    # Psi+, +- and -+ both announce Psi-, so the second and third terms
    # vanish. In the surviving two the environment contributes the overlap
    # f = exp(-4 mu (1 - sqrt(eta))) of the lost light, and the detected mode
    # contributes <beta|(1 - |0><0|)|-beta> = exp(-2|beta|^2) - exp(-|beta|^2)
    # which is about -|beta|^2 = -2 mu sqrt(eta), NEGATIVE, because the click
    # projects out vacuum. So the bracket is about -2 P_pass f and
    # e_ph = (1 - f)/2. A minus sign here gives (1 + f)/2 and, together with
    # e_ph <= 1/2, makes the program infeasible at every intensity.
    bracket = (ent(Z_P, s_pp, Z_P, s_mm)
               - ent(Z_M, s_pp, Z_M, s_mm)
               - ent(Z_P, s_pm, Z_P, s_mp)
               + ent(Z_M, s_pm, Z_M, s_mp))

    # Optimise the bracket rather than e_ph. The two differ by the affine map
    # e_ph = 1/2 + Re[bracket] / (4 P_pass), so the maximisers coincide, but
    # that map carries a factor 1/(4 P_pass) which is 250 at 30 dB and grows
    # as 1/sqrt(eta). Putting it in the objective and in the cap hands the
    # solver coefficients that blow up exactly where the problem is already
    # badly scaled. Here every coefficient is order one and e_ph is recovered
    # afterwards. The cap e_ph <= 1/2 becomes Re[bracket] <= 0.
    obj = cp.real(bracket)
    cons.append(obj <= 0.0)

    prob = cp.Problem(cp.Maximize(obj), cons)
    if solver is None:
        try:
            import mosek
            solver, opts = cp.MOSEK, {}
        except ImportError:
            # SCS at a tight tolerance does not converge on this problem.
            # CLARABEL works but warns about accuracy at high loss, where
            # the Psi-block entries are O(P_pass) against fail-block entries
            # and overlap constraints that are O(1). MOSEK is strongly
            # preferred; see the note on conditioning at the top of the file.
            solver, opts = cp.CLARABEL, {}
    else:
        opts = {}
    if solver_opts:
        opts = {**opts, **solver_opts}
    # mosek_params is rejected by every other solver, and a rejected kwarg
    # raises, which a caller catching exceptions would read as infeasibility.
    if solver is not cp.MOSEK:
        opts.pop("mosek_params", None)
    prob.solve(solver=solver, verbose=verbose, **opts)

    # Report the status. Returning 0.5 on a failed solve is indistinguishable
    # from a genuine e_ph = 0.5, and at high loss the two are easy to confuse:
    # P_pass is ~1e-3 by 30 dB, so the Psi blocks scale with it while the fail
    # block and every overlap constraint are O(1).
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return 0.5, e_bit, P_pass_key, prob.status
    e_ph = 0.5 + prob.value / (4.0 * P_pass_key)
    return float(np.clip(e_ph, 0.0, 0.5)), e_bit, P_pass_key, prob.status


def key_rate(M, mus, eta, p_dc=0.0, e_ali=0.0):
    e_ph, e_bit, P_pass, _ = solve_eph(M, mus, eta, p_dc, e_ali)
    return P_pass * max(0.0, 1.0 - h2(e_ph) - F_EC * h2(e_bit)), e_ph, e_bit


def eph_infinite(mu, eta):
    """
    Phase-error rate with infinitely many test states, from the argument of
    h2 in their Eq. 15 (loss only):

        e_ph = [1 - exp(-4 mu (1 - sqrt(eta))) exp(-2 mu sqrt(eta))] / 2

    The SDP maximises over Eve, and more test states constrain her further,
    so a finite-test-state result must sit at or above this. A row below it
    means the constraint set is wrong.
    """
    r = math.sqrt(eta)
    return 0.5 * (1.0 - math.exp(-4.0 * mu * (1.0 - r)) * math.exp(-2.0 * mu * r))


def rate_infinite(mu, eta):
    """Their Eq. 15: the loss-only rate with infinitely many test states."""
    return (1.0 - math.exp(-2.0 * mu * math.sqrt(eta))) * max(
        0.0, 1.0 - h2(eph_infinite(mu, eta)))


def plob(eta):
    # Undefined at eta = 1 (zero loss), where the bound is infinite.
    if eta >= 1.0:
        return float("inf")
    return -math.log2(1.0 - eta)


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
# The paper optimises mu_x per basis by coarse-grained exhaustive search.
# Start with a common mu; then allow the key basis and the test bases to
# differ, which is where the gain from more test states comes from.

# --------------------------------------------------------------------------
# Parallel grid
# --------------------------------------------------------------------------
# Every (loss, mu_key, mu_test) point is an independent solve, so the grid
# parallelises with no coordination. Each worker is pinned to a single solver
# thread: MOSEK is internally multithreaded and would otherwise oversubscribe
# the machine once several workers run at once. These problems are small
# (dim G = 3 x 4M^2), so one thread per solve and many solves in flight beats
# many threads per solve.

SOLVE_TIME_LIMIT = 180.0   # seconds per solve


def _one_point(task):
    Mv, dB, mk, mt = task
    eta = 10 ** (-dB / 10.0)
    mus = [mk] + [mt] * (Mv - 1)
    try:
        e_ph, e_bit, P_pass, status = solve_eph(
            Mv, mus, eta, P_DC, E_ALI,
            solver_opts={"mosek_params": {
                "MSK_IPAR_NUM_THREADS": 1,
                # Cap a single solve. Without it one pathological point
                # stalls a worker indefinitely and the run looks hung.
                "MSK_DPAR_OPTIMIZER_MAX_TIME": SOLVE_TIME_LIMIT,
            }},
        )
    except Exception as exc:
        return Mv, dB, mk, mt, None, f"exception: {type(exc).__name__}"
    R = P_pass * max(0.0, 1.0 - h2(e_ph) - F_EC * h2(e_bit))
    return Mv, dB, mk, mt, (R, e_ph, P_pass), status


if __name__ == "__main__":
    import multiprocessing as mp

    # M = 2, 3, 4 are the two, four and six test-state curves of Fig. 6(a).
    #
    # Caveat on the intensities. The paper optimises the intensity of each
    # test state; this scans a single mu_test shared by all M-1 test bases.
    # That is a restriction, so it can only lower the rate. If M = 3 or 4
    # comes out at or below M = 2, an independent intensity per test basis is
    # the first thing to try before suspecting a bug: replace the mt loop
    # with itertools.product(mu_test_grid, repeat=M-1) and pass the tuple as
    # the tail of mus. The cost is a grid of size |mu_test|^(M-1).
    #
    # Sizes, before the zero-norm reduction: 4 M^2 joint states and
    # dim G = 3 x 4 M^2, so 48, 108 and 192. The M = 4 grid is coarser to
    # keep the total tractable; widen it once you know where the optimum is.

    M_LIST = [2, 3, 4]
    losses = np.arange(5.0, 85.0, 5.0)

    # Workers, not cores. See the note by n_proc below: 3 is safe on a
    # laptop, and M = 4 is the memory-hungry case.
    N_PROC = 4

    # Coarse by design. The M = 2 run parked the optimum at mu_test = 0.126
    # or 0.251 at every distance above 25 dB, and the curves are straight
    # lines on a log plot, so a fine grid buys nothing. Widen once you know
    # where the optimum sits.
    GRIDS = {
        2: (np.logspace(-3.0, -0.3, 10), np.logspace(-2.0, 0.3, 8)),
        3: (np.logspace(-3.0, -0.3, 10), np.logspace(-2.0, 0.3, 8)),
        4: (np.logspace(-3.0, -0.3, 10), np.logspace(-2.0, 0.3, 8)),
    }

    tasks = []
    for Mv in M_LIST:
        mk_grid, mt_grid = GRIDS[Mv]
        tasks += [(Mv, dB, mk, mt)
                  for dB in losses for mk in mk_grid for mt in mt_grid]

    # Longest first. Cost grows steeply with M (dim G = 3 x 4 M^2, so 48,
    # 108, 192), and dispatching in M order means every cheap solve finishes
    # before any expensive one starts, which both wastes the tail of the run
    # on one straggler and makes progress look stalled at the boundary.
    tasks.sort(key=lambda t: -t[0])

    # Memory, not cores, is the binding constraint. Each worker holds a
    # canonicalised SDP whose size grows with M (dim G is 40, 96 and 176
    # after the zero-norm reduction), so oversubscribing the machine with
    # cpu_count() - 1 workers is what exhausts RAM on a laptop. Set N_PROC
    # by hand; raise it only if memory allows.
    n_proc = min(N_PROC, mp.cpu_count())
    print(f"{'loss only' if LOSS_ONLY else 'Parameter 2'}   "
          f"per-basis intensity   M = {M_LIST}")
    print(f"{len(tasks)} solves on {n_proc} processes", flush=True)

    t0 = time.time()
    best = {(Mv, dB): (0.0, None) for Mv in M_LIST for dB in losses}
    stats = {(Mv, dB): {} for Mv in M_LIST for dB in losses}
    done = 0
    # maxtasksperchild recycles workers so memory cvxpy does not release
    # between solves is reclaimed rather than accumulating over the run.
    with mp.Pool(processes=n_proc, maxtasksperchild=25) as pool:
        for Mv, dB, mk, mt, res, status in pool.imap_unordered(_one_point,
                                                              tasks,
                                                              chunksize=1):
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                rate = done / el
                eta_s = (len(tasks) - done) / rate if rate > 0 else 0.0
                print(f"  {done}/{len(tasks)}  elapsed {el/60:.1f} min  "
                      f"remaining ~{eta_s/60:.1f} min  (last M={Mv})",
                      flush=True)
            key = (Mv, dB)
            stats[key][status] = stats[key].get(status, 0) + 1
            if res is None:
                continue
            R, e_ph, P_pass = res
            if R > best[key][0]:
                best[key] = (R, (mk, mt, e_ph, P_pass))

    for Mv in M_LIST:
        mk_grid, _ = GRIDS[Mv]
        print()
        print(f"M = {Mv}  ({2*(Mv-1)} test states)")
        print(f"{'loss(dB)':>9} {'mu_key':>9} {'mu_test':>9} {'e_ph':>8} "
              f"{'eph_inf':>8} {'P_pass':>11} {'R':>11} {'R_inf':>11} "
              f"{'PLOB':>11} {'R/PLOB':>8}  solver status")
        for dB in losses:
            R, arg = best[(Mv, dB)]
            tally = ", ".join(f"{k}: {v}"
                              for k, v in sorted(stats[(Mv, dB)].items()))
            if arg is None:
                print(f"{dB:9.1f}   no feasible point{'':>74}  {tally}")
                continue
            mk, mt, e_ph, P_pass = arg
            eta = 10 ** (-dB / 10.0)
            inf_eph = eph_infinite(mk, eta)
            inf_R = max(rate_infinite(mu, eta) for mu in mk_grid)
            C = plob(eta)
            print(f"{dB:9.1f} {mk:9.5f} {mt:9.5f} {e_ph:8.4f} {inf_eph:8.4f} "
                  f"{P_pass:11.3e} {R:11.3e} {inf_R:11.3e} {C:11.3e} "
                  f"{R/C:8.3f}  {tally}")

    # ----------------------------------------------------------------------
    # Plot, laid out like their Fig. 6(a): log rate against total loss in dB,
    # PLOB red dashed, infinite test states black dotted, the finite-test
    # curves solid in the paper's colour order (two blue, four orange, six
    # green). Rates of zero are dropped rather than plotted at the axis
    # floor, so a curve that stops short means the bound went vacuous there.
    # ----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOURS = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    ax.plot(losses, [plob(10 ** (-dB / 10.0)) for dB in losses],
            "r--", lw=1.8, label="PLOB bound")

    mu_ref = np.logspace(-3.5, -0.3, 60)
    ax.plot(losses,
            [max(rate_infinite(mu, 10 ** (-dB / 10.0)) for mu in mu_ref)
             for dB in losses],
            "k:", lw=1.8, label="infinite test states")

    for Mv in M_LIST:
        xs, ys = [], []
        for dB in losses:
            R, arg = best[(Mv, dB)]
            if arg is None or R <= 0.0:
                continue
            xs.append(dB)
            ys.append(R)
        if xs:
            ax.plot(xs, ys, "-", lw=1.8, color=COLOURS.get(Mv),
                    label=f"{2*(Mv-1)} test states")

    ax.set_yscale("log")
    ax.set_xlabel("total loss (dB) between Alice and Bob")
    ax.set_ylabel("secret key rate (bits/channel use)")
    ax.set_xlim(0, losses.max())
    ax.set_ylim(1e-8, 1e0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.set_title("Phase-matching MDI-QKD, "
                 + ("loss only" if LOSS_ONLY else "Parameter 2"))
    fig.tight_layout()

    out = "fig6-reproduction.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")