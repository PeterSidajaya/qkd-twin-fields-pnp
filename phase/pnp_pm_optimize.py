"""Parallel black-box optimization of Stage 5A--PM test-basis transmissions.

Run from the phase/ directory or with the repository's phase/ directory in place:
    python pnp_pm_optimize.py --distances 0 20 40 --workers 4

Each candidate solves the unchanged pnp-stage-5a-PM.py Gram SDP in an
independent process.  Lambda_0 = 1 (key basis) is fixed; only lambda_1,...
are optimized.  A seeded Latin-hypercube exploration is followed by batched
coordinate refinements.  This is a heuristic, NOT a proof of global optimality.

The default coherent-reference statistics and P_monitor=0.99 remain synthetic
benchmark inputs, NOT calibrated monitor acceptance or certified key rates.
An SDP reported as 'optimal_inaccurate' is discarded, never promoted to a win.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


@lru_cache(maxsize=1)
def _pm_module():
    """Import original PM script once per process, without modifying it."""
    path = Path(__file__).resolve().with_name("pnp-stage-5a-PM.py")
    spec = importlib.util.spec_from_file_location("pnp_stage5a_pm_for_opt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latin_hypercube(n: int, dimension: int, lo: float, rng: random.Random):
    """Reproducible stratified candidates without any optimization dependency."""
    if n < 0 or dimension < 1 or not 0.0 <= lo < 1.0:
        raise ValueError("Invalid Latin-hypercube arguments")
    axes = []
    for _ in range(dimension):
        values = [lo + (1.0 - lo) * (k + rng.random()) / n for k in range(n)] if n else []
        rng.shuffle(values)
        axes.append(values)
    return [tuple(axes[j][i] for j in range(dimension)) for i in range(n)]


def _key(test_transmissions):
    return tuple(round(float(x), 7) for x in test_transmissions)


def _evaluate(task):
    """Top-level pickleable worker: one SDP solve with no nested parallelism."""
    distance, test, config = task
    started = time.monotonic()
    record = {
        "distance_km": float(distance),
        "lambda_by_basis": [1.0, *test],
        "status": "error",
        "e_ph": None,
        "e_bit": None,
        "p_pass": None,
        "R_acc": None,
        "R_original": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        # Set thread limits before importing numerical packages in each worker.
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        import cvxpy as cp
        pm = _pm_module()
        settings = pm.make_pm_settings(record["lambda_by_basis"])
        solver = None if config["solver"] == "auto" else getattr(cp, config["solver"].upper())
        result, _ = pm.solve_pm_ideal_benchmark(
            settings=settings,
            n_bar_before_loss=config["n_bar"],
            distance_km=distance,
            tau_s=config["tau_s"],
            c_max=config["c_max"],
            n_plus=config["n_plus"],
            p_mon_both_lower=config["p_mon"],
            loss_only=config["loss_only"],
            solver=solver,
        )
        record.update(
            status=str(result.status),
            e_ph=float(result.e_ph_upper),
            e_bit=float(result.e_bit),
            p_pass=float(result.p_pass_mon),
            R_acc=float(result.r_acc),
            R_original=None if result.r_original is None else float(result.r_original),
        )
        if any(not math.isfinite(record[k]) for k in ("e_ph", "e_bit", "p_pass", "R_acc")):
            record["status"] = "nonfinite"
        if record["status"] != "optimal":
            record["error"] = "Not an accepted optimum; requires solver verification"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    return record


def _rank(record):
    """Maximize accepted-round rate; minimize phase error to break zero-rate ties."""
    if record["status"] != "optimal":
        return (-math.inf, -math.inf)
    return (record["R_acc"], -record["e_ph"])


def _seed_profiles(dimension, lo):
    """Always include fixed, the original hand-picked PM profile and low loss."""
    fixed = (1.0,) * dimension
    if dimension == 3:
        handpicked = (0.8, 0.5, 0.2)
    else:
        handpicked = tuple(1.0 - 0.8 * (i + 1) / dimension for i in range(dimension))
    handpicked = tuple(max(lo, value) for value in handpicked)
    return [fixed, handpicked, (lo,) * dimension, ((1.0 + lo) / 2,) * dimension]


def optimize_distance(executor, distance, config, *, random_samples, refine_rounds, rng):
    """Parallel batch search, with exact-evaluation caching within this distance."""
    dimension = config["M"] - 1
    lo = config["lambda_min"]
    seen = {}

    def evaluate_batch(points):
        pending = {}
        for point in points:
            point = _key(tuple(min(1.0, max(lo, value)) for value in point))
            if point not in seen and point not in pending:
                pending[point] = executor.submit(_evaluate, (distance, point, config))
        by_future = {future: key for key, future in pending.items()}
        for future in as_completed(by_future):
            key = by_future[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"distance_km": distance, "lambda_by_basis": [1.0, *key],
                          "status": "worker_error", "R_acc": None, "e_ph": None,
                          "error": f"{type(exc).__name__}: {exc}"[:500]}
            seen[key] = result
        return len(pending)

    initial = _seed_profiles(dimension, lo)
    initial += latin_hypercube(random_samples, dimension, lo, rng)
    evaluated = evaluate_batch(initial)
    valid = [row for row in seen.values() if row["status"] == "optimal"]
    if not valid:
        raise RuntimeError(f"Distance {distance:g} km: no solver-certified 'optimal' status. "
                           "Try --solver mosek or inspect the saved solver diagnostics.")
    best = max(valid, key=_rank)
    step = 0.25 * (1.0 - lo)
    for _ in range(refine_rounds):
        test = best["lambda_by_basis"][1:]
        neighbors = []
        for j in range(dimension):
            for direction in (-1, 1):
                neighbor = list(test)
                neighbor[j] += direction * step
                neighbors.append(tuple(neighbor))
        evaluated += evaluate_batch(neighbors)
        candidate = max((r for r in seen.values() if r["status"] == "optimal"), key=_rank)
        if _rank(candidate) > _rank(best):
            best = candidate
        else:
            step *= 0.5
        if step < 0.002:
            break
    fixed = seen[_key((1.0,) * dimension)]
    return best, fixed, list(seen.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distances", type=float, nargs="+", default=[0.0, 20.0, 40.0])
    parser.add_argument("--M", type=int, default=4, help="Number of bases, including key basis")
    parser.add_argument("--n-bar", type=float, default=0.01, help="Input intensity before trusted basis loss")
    parser.add_argument("--tau-s", type=float, default=1e-8)
    parser.add_argument("--c-max", type=int, default=1_000_000)
    parser.add_argument("--n-plus", type=int, default=3)
    parser.add_argument("--p-mon", type=float, default=0.99,
                        help="Externally justified monitor-acceptance lower bound; 0.99 is placeholder")
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=18, help="Global Latin-hypercube samples per distance")
    parser.add_argument("--refine-rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--solver", choices=["auto", "mosek", "clarabel"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stage5a-PM-opt-results.json"))
    parser.add_argument("--plot", action="store_true", help="Plot optimized and fixed profiles")
    args = parser.parse_args(argv)
    if not 2 <= args.M or not 0 <= args.lambda_min < 1:
        parser.error("Require M >= 2 and 0 <= --lambda-min < 1")
    if args.workers < 1 or args.samples < 0 or args.refine_rounds < 0:
        parser.error("workers >= 1, samples >= 0 and refine-rounds >= 0 required")
    if not 0 < args.p_mon <= 1 or not 0 <= args.tau_s <= 1 or args.n_bar <= 0:
        parser.error("Invalid monitor or mean-energy parameters")
    if any(d < 0 for d in args.distances):
        parser.error("Distances must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = {"M": args.M, "n_bar": args.n_bar, "tau_s": args.tau_s,
              "c_max": args.c_max, "n_plus": args.n_plus, "p_mon": args.p_mon,
              "loss_only": args.loss_only, "lambda_min": args.lambda_min,
              "solver": args.solver}
    rng = random.Random(args.seed)
    data = {"config": {**config, "distances_km": args.distances, "samples": args.samples,
                       "refine_rounds": args.refine_rounds, "workers": args.workers,
                       "seed": args.seed},
            "warning": "Synthetic coherent-reference benchmark; p_mon is externally supplied. "
                       "A numerical SDP optimum is not a rigorous dual certificate. "
                       "Parallel search is heuristic, not globally optimal.",
            "best_by_distance": [], "evaluations": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Optimizing Stage 5A--PM test transmissions (lambda_0=1; key unchanged).", flush=True)
    print("WARNING: coherent-reference synthetic statistics and illustrative acceptance bound.", flush=True)
    plot_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for distance in args.distances:
            best, fixed, evaluated = optimize_distance(
                pool, distance, config, random_samples=args.samples,
                refine_rounds=args.refine_rounds, rng=rng,
            )
            data["evaluations"].extend(evaluated)
            data["best_by_distance"].append({"distance_km": distance, "optimized": best,
                                             "fixed": fixed, "n_evaluated": len(evaluated)})
            for name, record in (("fixed (Stage5A)", fixed), ("optimized basis-loss PM", best)):
                plot_rows.append({"profile": name, "L_km": distance,
                                  "e_ph": record["e_ph"], "R_original": record.get("R_original")})
            print(f"{distance:g} km: {len(evaluated)} evaluations; fixed R_acc={fixed.get('R_acc')}; "
                  f"best R_acc={best['R_acc']:.6g}, e_ph={best['e_ph']:.6g}, "
                  f"lambda={best['lambda_by_basis']}", flush=True)
            # Checkpoint completed distances even if a later distance fails.
            args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if args.plot:
        pm = _pm_module()
        pm.plot_pm_profiles(plot_rows, output_dir=args.output.parent / "stage5a-PM-opt-plots")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
