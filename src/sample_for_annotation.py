#!/usr/bin/env python
"""
Draw a stratified sample of judged rows for human validation of the judge.

Stratifies by (model, judge label) so that rare cells (e.g. LAT answers,
baseline refusals) are represented, then shuffles and hides the judge's
verdict in a separate column block so annotators can label blind.

Output columns:
    sample_id, model, id, category, prompt, response,
    human_label   <- annotator fills in: direct_answer / direct_refusal / indirect_refusal
    (judge_label, judge_raw are appended at the far right; hide these columns while annotating)

Afterwards compute agreement with:
    python sample_for_annotation.py --score annotation_sample.csv
"""

import argparse
import sys

import pandas as pd

LABELS = ("direct_answer", "direct_refusal", "indirect_refusal")


def draw(args):
    frames = []
    for path, name in zip(args.judged, args.names):
        df = pd.read_csv(path)
        df = df[df["label"].isin(LABELS)].copy()
        df["model"] = name
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)

    parts = []
    for (model, label), grp in allrows.groupby(["model", "label"]):
        k = min(args.n_per_cell, len(grp))
        parts.append(grp.sample(k, random_state=args.seed))
        print(f"  {model:9s} {label:17s} pool={len(grp):5d} sampled={k}")
    sample = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))

    out = sample[["sample_id", "model", "id", "category", "prompt", "response"]].copy()
    out["human_label"] = ""
    out["judge_label"] = sample["label"].values
    out["judge_raw"] = sample["judge_raw"].values
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows to {args.out}")
    print("Annotate the 'human_label' column blind (hide judge_label / judge_raw), then run --score.")


def score(path):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    df = pd.read_csv(path)
    df = df[df["human_label"].isin(LABELS)]
    if df.empty:
        sys.exit("No rows with a valid human_label yet.")
    y_h, y_j = df["human_label"], df["judge_label"]

    print(f"Annotated rows: {len(df)}")
    print(f"Exact agreement (3-way): {(y_h == y_j).mean():.3f}")
    print(f"Cohen's kappa (3-way):   {cohen_kappa_score(y_h, y_j):.3f}")

    # Binary refuse-vs-answer, which is what the headline metric uses
    bh = y_h != "direct_answer"
    bj = y_j != "direct_answer"
    print(f"Agreement (refuse/answer): {(bh == bj).mean():.3f}")
    print(f"Cohen's kappa (refuse/answer): {cohen_kappa_score(bh, bj):.3f}")

    print("\nConfusion matrix (rows = human, cols = judge):")
    cm = confusion_matrix(y_h, y_j, labels=list(LABELS))
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    print("\nPer-model agreement (refuse/answer):")
    print(df.assign(agree=(bh == bj).values).groupby("model")["agree"].mean().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="*")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--n-per-cell", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--score", help="path to an annotated sample CSV")
    args = ap.parse_args()

    if args.score:
        score(args.score)
    else:
        if not (args.judged and args.names and args.out):
            sys.exit("need --judged --names --out (or --score FILE)")
        assert len(args.judged) == len(args.names)
        draw(args)


if __name__ == "__main__":
    main()
