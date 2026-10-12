#!/usr/bin/env python
"""
Aggregate judged OR-Bench results across alignment methods.

Produces:
  - overall refusal rate per model with 95% bootstrap CI
  - breakdown into direct / indirect refusal
  - per-category refusal rates
  - paired McNemar test of each aligned model vs. the baseline
    (same prompts -> paired design; this is the right test, not a two-proportion z-test)
  - a LaTeX table you can paste into the paper

Usage:
    python summarize_orbench.py \
        --judged outputs/orbench/judged/hard1k_baseline.judged.csv \
                 outputs/orbench/judged/hard1k_sft.judged.csv \
                 outputs/orbench/judged/hard1k_dpo.judged.csv \
                 outputs/orbench/judged/hard1k_lat.judged.csv \
        --names baseline sft dpo lat \
        --baseline baseline \
        --out outputs/orbench/judged/summary
"""

import argparse
import os

import numpy as np
import pandas as pd

REFUSAL = ("direct_refusal", "indirect_refusal")
VALID = ("direct_answer",) + REFUSAL


def bootstrap_ci(x, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def mcnemar(a, b):
    """Exact McNemar test on paired binary arrays. Returns (b01, b10, p)."""
    from scipy.stats import binomtest
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    b01 = int(((~a) & b).sum())   # baseline answered, model refused
    b10 = int((a & (~b)).sum())   # baseline refused, model answered
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n > 0 else 1.0
    return b01, b10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    assert len(args.judged) == len(args.names)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    frames = {}
    for path, name in zip(args.judged, args.names):
        df = pd.read_csv(path)
        n_total = len(df)
        df = df[df["label"].isin(VALID)].copy()
        dropped = n_total - len(df)
        if dropped:
            print(f"WARNING [{name}]: dropping {dropped} rows with label not in {VALID}")
        df["refused"] = df["label"].isin(REFUSAL)
        df["direct"] = df["label"] == "direct_refusal"
        df["indirect"] = df["label"] == "indirect_refusal"
        frames[name] = df.set_index("id").sort_index()

    # Align on common ids so comparisons are paired.
    common = set.intersection(*(set(f.index) for f in frames.values()))
    for name in frames:
        frames[name] = frames[name].loc[sorted(common)]
    print(f"Paired on {len(common)} common prompts.\n")

    # ---------------- overall ----------------
    rows = []
    base = frames.get(args.baseline)
    for name, df in frames.items():
        lo, hi = bootstrap_ci(df["refused"].values)
        row = {
            "model": name,
            "n": len(df),
            "refusal_rate": df["refused"].mean(),
            "ci_lo": lo,
            "ci_hi": hi,
            "direct_refusal": df["direct"].mean(),
            "indirect_refusal": df["indirect"].mean(),
        }
        if base is not None and name != args.baseline:
            b01, b10, p = mcnemar(base["refused"].values, df["refused"].values)
            row.update({"vs_base_delta": row["refusal_rate"] - base["refused"].mean(),
                        "newly_refused": b01, "newly_answered": b10, "mcnemar_p": p})
        rows.append(row)
    overall = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== Overall over-refusal on OR-Bench-Hard-1K (lower is better) ===")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    overall.to_csv(f"{args.out}_overall.csv", index=False)

    # ---------------- per category ----------------
    cat = pd.DataFrame({
        name: df.groupby("category")["refused"].mean() for name, df in frames.items()
    })
    cat["n"] = next(iter(frames.values())).groupby("category").size()
    print("\n=== Refusal rate by category ===")
    print(cat.to_string(float_format=lambda v: f"{v:.3f}"))
    cat.to_csv(f"{args.out}_by_category.csv")

    # ---------------- LaTeX ----------------
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Refusal rate (\%) $\downarrow$ & Direct (\%) & Indirect (\%) \\",
        r"\midrule",
    ]
    for _, r in overall.iterrows():
        lines.append(
            f"{r['model']} & {100*r['refusal_rate']:.1f} "
            f"[{100*r['ci_lo']:.1f}, {100*r['ci_hi']:.1f}] & "
            f"{100*r['direct_refusal']:.1f} & {100*r['indirect_refusal']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{args.out}_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}_overall.csv, {args.out}_by_category.csv, {args.out}_table.tex")


if __name__ == "__main__":
    main()
