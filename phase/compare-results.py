"""
Combine the standardized CSV outputs of all PnP stage scripts.

Usage:
    python phase/compare-results.py

Reads phase/results/{stage}/results.csv if present, writes one combined CSV
phase/results/all-stages.csv, and compares the fixed PM lambda=1 profile to
Stage5A at N_plus=3 (when both files exist).

Does NOT recompute any SDP or assert security equivalence across stages.
"""
from __future__ import annotations

import csv
from pathlib import Path

from pnp_results import COLUMNS

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
STAGES=("stage1","stage2","stage3","stage4",
        "stage5a","stage5a-PM","stage5b")


def read_stage(stage):
    path=ROOT/stage/"results.csv"
    if not path.is_file():
        print(f"Not yet generated: {path}")
        return []
    with path.open(newline="",encoding="utf-8") as fh:
        reader=csv.DictReader(fh)
        if list(reader.fieldnames or ())!=list(COLUMNS):
            raise ValueError(f"Unexpected schema in {path}")
        return list(reader)


def combined_table():
    rows=[]
    groups={}
    for stage in STAGES:
        groups[stage]=read_stage(stage)
        rows.extend(groups[stage])
    if not rows:
        print("No results found. First run the stage scripts.")
        return []

    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/"all-stages.csv"
    with path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Combined {len(rows)} rows in {path}")
    return groups


def compare_pm_fixed_limit(groups):
    original={
        float(r["L_km"]):r
        for r in groups["stage5a"]
        if r["profile"]=="N_plus=3" and r["status"].startswith("optimal")
    }
    pm={
        float(r["L_km"]):r
        for r in groups["stage5a-PM"]
        if r["profile"]=="fixed (Stage5A)" and r["status"].startswith("optimal")
    }
    shared=sorted(set(original).intersection(pm))
    if not shared:
        print("No matching Stage5A N_plus=3 / PM fixed-profile rows yet.")
        return

    print("\nFixed-loss PM vs original Stage5A, matching total distance:")
    print(f"{'L_km':>7} {'delta_eph':>13} {'delta_e_bit':>13} {'delta_Racc':>13}")
    for L in shared:
        a,b=original[L],pm[L]
        deviations=[]
        for key in ("e_ph_upper","e_bit","R_acc"):
            deviations.append(float(b[key])-float(a[key]))
        print(f"{L:7.1f}"+ "".join(f" {d:13.5e}" for d in deviations))
    print("SDP solver tolerances can cause small nonzero differences; "
          "this compares saved benchmark values, not actual attacks.")


def main():
    groups=combined_table()
    if groups:
        compare_pm_fixed_limit(groups)


if __name__=="__main__":
    main()
