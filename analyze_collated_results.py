#!/usr/bin/env python3
"""Read collated_results.xlsx and produce a comprehensive top-N analysis text.

The .xlsx is expected to be one concatenated table where each row is a single
(configuration, variant) result -- a stack of summary1.csv (NN) and
summary2.csv (XGB) files across all the runs you did. Configuration columns
(allstar, augs, text, test_set, head) carry the run dimensions; metric
columns carry test_best_f1, test_auc, ind_f1, php_f1, recall@p80, etc.

Run it locally and paste the resulting .txt back to Claude for analysis.

Defaults:
  --in   collated_results.xlsx
  --out  results_report.txt
  --top_n 10
  --axes  auto-detected (string-typed columns with <=10 unique values
          excluding 'variant'). Override with --axes head,allstar,augs,text,test_set
          if auto-detection picks the wrong columns.

What the report contains (every section is plain text; designed to be
copy-pasted whole into a chat reply):

  1. AXIS SUMMARY                 -- what was tested and how often
  2. SCHEMA / NUMERIC METRICS     -- which metric columns the file has
  3. TOP-N OVERALL                -- by test_best_f1, test_auc, test_ap
  4. TOP-N PER REGION             -- by ind_f1 and php_f1
  5. TOP-N BY RECALL@P            -- p80/p85/p90/p95
  6. TOP-N PER REGION RECALL@P    -- ind_r@p80/85/90/95, php_r@p80/85/90/95
  7. BOTTOM-5                     -- sanity check (worst performers)
  8. PER-AXIS SENSITIVITY         -- mean+std test_best_f1 grouped by each axis
  9. 2-AXIS CROSS-TABS            -- pivot mean test_best_f1 over each pair
 10. VARIANT WIN COUNT            -- how often each variant lands in top-5
                                     within a (axes-fixed) group
 11. VARIANT STABILITY            -- mean/std/min/max test_best_f1 per variant
 12. BEST PER (head, test_set)    -- top 3 within each evaluation regime
 13. REGION GAP                   -- |ind_f1 - php_f1| smallest (balanced) /
                                     largest (uneven)

  -- recipe-focused sections (require a 'recipe' column; from COMMANDSIMP collation) --

 14. DELTA VS BASELINE recipe     -- per (head, variant): metric delta of every
                                     non-baseline recipe vs --baseline_recipe.
                                     This is THE table for "did R1..R5 help?"
 15. RECIPE x VARIANT PIVOT       -- heatmap-style (mean test_best_f1) per head
 16. BEST VARIANT PER (head,recipe) -- top 3 per (head, recipe), full metrics
 17. ARCH FAMILY x RECIPE         -- linear vs tiny vs default sanity check;
                                     overfitting indicator
 18. CALIBRATION FOCUS            -- mean high-precision recall per recipe;
                                     this is where per-client-std wins should
                                     show (AUC was already 0.92)
 19. PER-RECIPE REGION            -- IND vs PHP per recipe; was R5 worth it?

If your .xlsx has multiple sheets, pass --sheet <name_or_index>.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# Columns we will rank by (in priority order) -- only those present in the file
# actually get a section.
TOP_METRICS_OVERALL = ["test_best_f1", "test_auc", "test_ap", "test_f1@0.5",
                       "test_topk_f1"]
REGION_METRICS = ["ind_f1", "php_f1"]
RECALL_METRICS = ["recall@p50", "recall@p80", "recall@p85", "recall@p90",
                  "recall@p95"]
REGION_RECALL_METRICS = [f"{r}_r@p{p}" for r in ("ind", "php")
                         for p in (80, 85, 90, 95)]

# Recipe-focused analysis: metrics we want side-by-side per recipe.
RECIPE_METRICS = ["test_best_f1", "test_auc", "test_ap",
                  "recall@p85", "recall@p90", "recall@p95",
                  "ind_f1", "php_f1"]

# Arch families inferred from variant name prefix.
ARCH_FAMILIES = ("default", "tiny", "linear", "tierA_pick", "wavlm_xgb",
                 "whisper_xgb", "text_top20_xgb", "text_all_xgb",
                 "text_stylo_xgb", "everything_xgb")


def infer_arch_family(variant: str) -> str:
    if not isinstance(variant, str):
        return "other"
    for fam in ARCH_FAMILIES:
        if variant.startswith(fam):
            return fam
    return "other"


def detect_axes(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Treat any string-typed column with <=10 unique values as a config axis."""
    out = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if s.dtype == "object" and s.nunique(dropna=True) <= 10:
            out.append(col)
    return out


def display_cols(df: pd.DataFrame, axes: list[str]) -> list[str]:
    """Compact column list for top-N tables: variant + axes + a few metrics."""
    want = (["variant"] + axes +
            ["test_best_f1", "test_auc", "test_ap",
             "ind_f1", "php_f1",
             "recall@p80", "recall@p90"])
    return [c for c in want if c in df.columns]


def fmt_section(title: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n"


def fmt_subsection(title: str) -> str:
    return f"\n{'-' * 80}\n{title}\n{'-' * 80}\n"


def safe_df_string(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """to_string with NaN-safe formatting and an optional row cap."""
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_string(index=False, na_rep="NaN")


def write_top_n(out, df, by: str, n: int, cols: list[str]) -> None:
    if by not in df.columns:
        return
    sub = df.sort_values(by, ascending=False, na_position="last")
    out.write(fmt_section(f"TOP-{n} BY {by}"))
    out.write(safe_df_string(sub[cols], max_rows=n))
    out.write("\n")


def write_bottom_n(out, df, by: str, n: int, cols: list[str]) -> None:
    if by not in df.columns:
        return
    sub = df.sort_values(by, ascending=True, na_position="last")
    out.write(fmt_section(f"BOTTOM-{n} BY {by} (sanity check)"))
    out.write(safe_df_string(sub[cols], max_rows=n))
    out.write("\n")


def write_axis_sensitivity(out, df, axes: list[str], metric: str) -> None:
    if metric not in df.columns or not axes:
        return
    out.write(fmt_section(f"PER-AXIS SENSITIVITY (mean / std / min / max / n  on  {metric})"))
    for ax in axes:
        out.write(fmt_subsection(f"By {ax}"))
        g = (df.groupby(ax)[metric]
               .agg(mean="mean", std="std", min="min", max="max", n="count")
               .round(4)
               .sort_values("mean", ascending=False))
        out.write(g.to_string())
        out.write("\n")


def write_2axis_crosstabs(out, df, axes: list[str], metric: str) -> None:
    if metric not in df.columns or len(axes) < 2:
        return
    out.write(fmt_section(f"2-AXIS CROSS-TABS  (mean {metric})"))
    for ax1, ax2 in combinations(axes, 2):
        out.write(fmt_subsection(f"{ax1}  x  {ax2}"))
        tab = (df.pivot_table(values=metric, index=ax1, columns=ax2,
                              aggfunc="mean")
                 .round(4))
        out.write(tab.to_string(na_rep="NaN"))
        out.write("\n")
        # also count cells so we know how many observations are behind each mean
        cnt = (df.pivot_table(values=metric, index=ax1, columns=ax2,
                              aggfunc="count"))
        out.write("(n per cell)\n")
        out.write(cnt.fillna(0).astype(int).to_string())
        out.write("\n")


def write_variant_wins(out, df, axes: list[str], metric: str, top_n_in_group: int = 5) -> None:
    if metric not in df.columns or "variant" not in df.columns:
        return
    out.write(fmt_section(
        f"VARIANT WIN COUNT (times each variant lands in top-{top_n_in_group} "
        f"by {metric} within an axes-fixed group)"))
    if axes:
        wins: dict[str, int] = {}
        for _, sub in df.groupby(axes, dropna=False):
            top = sub.sort_values(metric, ascending=False).head(top_n_in_group)
            for v in top["variant"].tolist():
                wins[v] = wins.get(v, 0) + 1
        wdf = (pd.DataFrame(sorted(wins.items(), key=lambda x: -x[1]),
                            columns=["variant", "top_count"])
               .head(30))
        out.write(wdf.to_string(index=False))
    else:
        out.write("(no axes detected; skipping)")
    out.write("\n")


def write_variant_stability(out, df, metric: str) -> None:
    if metric not in df.columns or "variant" not in df.columns:
        return
    out.write(fmt_section(
        f"VARIANT STABILITY  (across all configurations, on {metric})"))
    g = (df.groupby("variant")[metric]
           .agg(mean="mean", std="std", min="min", max="max", n="count")
           .round(4)
           .sort_values("mean", ascending=False))
    out.write(g.head(40).to_string())
    out.write("\n")


def write_best_per_head_testset(out, df, cols: list[str]) -> None:
    if not {"head", "test_set"}.issubset(df.columns):
        return
    out.write(fmt_section("BEST per (head x test_set)"))
    for (h, ts), sub in df.groupby(["head", "test_set"], dropna=False):
        out.write(fmt_subsection(f"head={h}   test_set={ts}   n={len(sub)}"))
        top = sub.sort_values("test_best_f1", ascending=False).head(5)
        out.write(safe_df_string(top[cols]))
        out.write("\n")


def write_region_gap(out, df, axes: list[str]) -> None:
    if not {"ind_f1", "php_f1"}.issubset(df.columns):
        return
    out.write(fmt_section("REGION GAP  |ind_f1 - php_f1|"))
    df = df.copy()
    df["region_gap"] = (df["ind_f1"] - df["php_f1"]).abs()
    keep = ["variant"] + axes + ["ind_f1", "php_f1", "region_gap"]
    keep = [c for c in keep if c in df.columns]
    out.write(fmt_subsection("Most balanced (smallest gap)"))
    out.write(safe_df_string(df.nsmallest(10, "region_gap")[keep]))
    out.write("\n")
    out.write(fmt_subsection("Most uneven (biggest gap)"))
    out.write(safe_df_string(df.nlargest(10, "region_gap")[keep]))
    out.write("\n")


def write_quick_takeaways(out, df, axes: list[str], metric: str) -> None:
    """Tiny auto-summary at the top so the user (and Claude) sees the headlines
    before scrolling through 13 sections."""
    out.write(fmt_section("QUICK TAKEAWAYS"))
    if metric not in df.columns:
        out.write("(primary metric missing -- nothing to summarize)\n")
        return
    out.write(f"Primary metric: {metric}\n")
    out.write(f"Total rows analyzed: {len(df)}\n")
    out.write(f"Best single row:  ")
    top = df.sort_values(metric, ascending=False).head(1)
    if not top.empty:
        cols = [c for c in (["variant"] + axes + [metric, "test_auc", "ind_f1", "php_f1"])
                if c in df.columns]
        out.write(top[cols].to_string(index=False, na_rep="NaN"))
        out.write("\n")

    # Per-axis deltas (max minus min of the per-level mean)
    if axes:
        out.write("\nAxis impact (max - min of per-level mean):\n")
        rows = []
        for ax in axes:
            means = df.groupby(ax)[metric].mean()
            if len(means) >= 2:
                rows.append({"axis": ax,
                             "max_minus_min": round(means.max() - means.min(), 4),
                             "best_level": str(means.idxmax()),
                             "best_level_mean": round(means.max(), 4),
                             "worst_level": str(means.idxmin()),
                             "worst_level_mean": round(means.min(), 4),
                             "n_levels": int(len(means))})
        if rows:
            adf = (pd.DataFrame(rows).sort_values("max_minus_min", ascending=False))
            out.write(adf.to_string(index=False))
            out.write("\n")


def write_recipe_delta_vs_baseline(out, df, baseline: str, metrics: list[str]) -> None:
    """For each non-baseline recipe, per (head, variant): metric delta vs baseline.

    This is the *primary* table for evaluating whether per-client std / few-shot /
    augs actually moved the needle. The baseline row is the reference for that
    (head, variant). Shows top winners and losers per recipe."""
    if "recipe" not in df.columns or "variant" not in df.columns:
        return
    if baseline not in set(df["recipe"].dropna().astype(str)):
        out.write(fmt_section(
            f"14. DELTA VS BASELINE recipe='{baseline}'  (baseline missing -- skipping)"))
        return
    head_col = "head" if "head" in df.columns else None
    keys = ["variant"] + ([head_col] if head_col else [])
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    base = df[df["recipe"].astype(str) == baseline][keys + metrics]
    # if duplicates in baseline (e.g. multiple seeds), average them
    base = base.groupby(keys, dropna=False)[metrics].mean().reset_index()
    base = base.rename(columns={m: f"{m}_base" for m in metrics})

    out.write(fmt_section(
        f"14. DELTA VS BASELINE recipe='{baseline}'  (per head x variant; sorted by delta_test_best_f1)"))

    other_recipes = sorted(set(df["recipe"].astype(str)) - {baseline})
    for rec in other_recipes:
        sub = df[df["recipe"].astype(str) == rec][keys + metrics]
        sub = sub.groupby(keys, dropna=False)[metrics].mean().reset_index()
        merged = sub.merge(base, on=keys, how="inner")
        if merged.empty:
            out.write(fmt_subsection(f"recipe={rec}: no overlapping (head, variant) rows"))
            continue
        for m in metrics:
            merged[f"d_{m}"] = merged[m] - merged[f"{m}_base"]
        # display columns: keys + (metric, baseline, delta) for the primary metric,
        # then deltas for the rest
        primary = "test_best_f1" if "test_best_f1" in metrics else metrics[0]
        show_cols = keys + [primary, f"{primary}_base", f"d_{primary}"]
        for m in metrics:
            if m != primary:
                show_cols.append(f"d_{m}")
        out.write(fmt_subsection(
            f"recipe={rec}  vs  {baseline}   (n_overlap={len(merged)})"))
        sorted_df = merged.sort_values(f"d_{primary}", ascending=False).round(4)
        out.write("TOP-10 WINNERS:\n")
        out.write(safe_df_string(sorted_df[show_cols], max_rows=10))
        out.write("\n\nBOTTOM-5 LOSERS:\n")
        out.write(safe_df_string(sorted_df[show_cols].tail(5)))
        out.write("\n\nMEAN deltas across all variants:\n")
        delta_means = {m: round(merged[f"d_{m}"].mean(), 4) for m in metrics}
        out.write(pd.DataFrame([delta_means]).to_string(index=False))
        out.write("\n")


def write_recipe_variant_pivot(out, df, metric: str) -> None:
    """recipe x variant heatmap-style pivot. Separate pivot per head."""
    if "recipe" not in df.columns or "variant" not in df.columns:
        return
    if metric not in df.columns:
        return
    out.write(fmt_section(f"15. RECIPE x VARIANT PIVOT  (mean {metric}; one pivot per head)"))
    if "head" in df.columns:
        for h, sub in df.groupby("head", dropna=False):
            out.write(fmt_subsection(f"head={h}   n={len(sub)}"))
            tab = (sub.pivot_table(values=metric, index="variant",
                                    columns="recipe", aggfunc="mean")
                      .round(4))
            # sort variants by mean across recipes desc for readability
            tab = tab.assign(_mean=tab.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")
            out.write(tab.to_string(na_rep="NaN"))
            out.write("\n")
    else:
        tab = (df.pivot_table(values=metric, index="variant",
                              columns="recipe", aggfunc="mean")
                 .round(4))
        out.write(tab.to_string(na_rep="NaN"))
        out.write("\n")


def write_best_per_recipe(out, df, axes: list[str]) -> None:
    """Top 3 variants per (head, recipe) with full metric breakdown.
    This is the 'what would I deploy' number per recipe."""
    if "recipe" not in df.columns:
        return
    if "test_best_f1" not in df.columns:
        return
    out.write(fmt_section("16. BEST VARIANT PER (head, recipe)  -- top 3 each, full metrics"))
    keys = [c for c in ["head", "recipe"] if c in df.columns]
    metric_cols = [c for c in RECIPE_METRICS if c in df.columns]
    show_cols = (["variant"] + [a for a in axes if a not in keys and a in df.columns]
                 + metric_cols)
    for key_vals, sub in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        tag = "  ".join(f"{k}={v}" for k, v in zip(keys, key_vals))
        out.write(fmt_subsection(f"{tag}   n={len(sub)}"))
        top = sub.sort_values("test_best_f1", ascending=False).head(3)
        out.write(safe_df_string(top[[c for c in show_cols if c in top.columns]]))
        out.write("\n")


def write_arch_family_summary(out, df) -> None:
    """Mean F1 / recall@p85 per arch family per recipe.

    If 'linear' matches 'tiny': overfitting indicator (only-first-moment signal).
    If 'tiny' beats 'default': default is overfitting client-A artifacts.
    """
    if "variant" not in df.columns or "test_best_f1" not in df.columns:
        return
    df = df.copy()
    df["arch_family"] = df["variant"].apply(infer_arch_family)
    out.write(fmt_section("17. ARCH FAMILY x RECIPE  (linear vs tiny vs default -- overfitting check)"))
    metrics_show = [m for m in ("test_best_f1", "test_auc", "recall@p85", "recall@p90")
                    if m in df.columns]
    if not metrics_show:
        return
    group_cols = [c for c in ("head", "recipe") if c in df.columns] + ["arch_family"]
    for m in metrics_show:
        out.write(fmt_subsection(f"mean({m})  grouped by {group_cols}"))
        tab = (df.groupby(group_cols, dropna=False)[m]
                 .agg(mean="mean", n="count")
                 .round(4))
        out.write(tab.to_string())
        out.write("\n")


def write_calibration_focus(out, df) -> None:
    """audios6 calibration was the bottleneck (AUC=0.92, F1=0.61).

    Per recipe: mean recall@p85, p90, p95 across all variants -- shows whether
    per-client-std / few-shot improved CALIBRATION specifically (vs ranking,
    which was already fine)."""
    if "recipe" not in df.columns:
        return
    recall_cols = [c for c in ("recall@p80", "recall@p85", "recall@p90", "recall@p95")
                   if c in df.columns]
    if not recall_cols:
        return
    out.write(fmt_section(
        "18. CALIBRATION FOCUS  (mean high-precision recall per recipe; AUC was already 0.92 -- this is where wins should show)"))
    group_cols = [c for c in ("head", "recipe") if c in df.columns]
    tab = (df.groupby(group_cols, dropna=False)[recall_cols]
             .mean()
             .round(4))
    out.write(tab.to_string())
    out.write("\n")
    if "test_auc" in df.columns:
        out.write(fmt_subsection("companion: mean test_auc per recipe (ranking quality -- baseline 0.92)"))
        auc_tab = (df.groupby(group_cols, dropna=False)["test_auc"]
                     .mean().round(4))
        out.write(auc_tab.to_string())
        out.write("\n")


def write_region_per_recipe(out, df) -> None:
    """Per-recipe IND vs PHP performance.

    R0-R4 are IND-tested; R5 is PHP-tested. This table makes it explicit
    which recipe tested which region, and whether PHP-focused few-shot (R5)
    actually moved PHP."""
    if "recipe" not in df.columns:
        return
    keep = [c for c in ("ind_f1", "php_f1", "ind_r@p85", "php_r@p85",
                        "ind_r@p90", "php_r@p90", "test_region_filter")
            if c in df.columns]
    if not keep:
        return
    out.write(fmt_section("19. PER-RECIPE REGION PERFORMANCE  (IND vs PHP; was R5 worth it?)"))
    group_cols = [c for c in ("head", "recipe") if c in df.columns]
    metric_cols = [c for c in ("ind_f1", "php_f1", "ind_r@p85", "php_r@p85",
                                "ind_r@p90", "php_r@p90") if c in df.columns]
    if metric_cols:
        tab = (df.groupby(group_cols, dropna=False)[metric_cols]
                 .mean()
                 .round(4))
        out.write(tab.to_string(na_rep="NaN"))
        out.write("\n")
    if "test_region_filter" in df.columns:
        out.write(fmt_subsection("which region was each recipe tested on?"))
        rf = (df.groupby([c for c in ("recipe",) if c in df.columns])["test_region_filter"]
                .agg(lambda s: ",".join(sorted(set(map(str, s.dropna()))))))
        out.write(rf.to_string())
        out.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", default="collated_results.xlsx")
    ap.add_argument("--out", dest="output_path", default="results_report.txt")
    ap.add_argument("--axes", default="",
                    help="comma-separated config-axis columns; empty -> auto-detect")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--sheet", default="0",
                    help="sheet name or index (default 0 = first sheet)")
    ap.add_argument("--primary_metric", default="test_best_f1")
    ap.add_argument("--baseline_recipe", default="r0",
                    help="recipe name used as the reference in the delta tables "
                         "(default 'r0'). Set to '' to skip the delta section.")
    args = ap.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    if not in_path.exists():
        print(f"input file not found: {in_path}", file=sys.stderr)
        return 1

    sheet: str | int = args.sheet
    try:
        sheet = int(sheet)
    except ValueError:
        pass

    df = pd.read_excel(in_path, sheet_name=sheet)
    # Strip stray whitespace from column names (common in copy-paste collations).
    df.columns = [str(c).strip() for c in df.columns]
    n_rows, n_cols = df.shape
    print(f"Loaded {n_rows} rows x {n_cols} cols from {in_path}  (sheet={sheet})")

    # Pick axis columns
    if args.axes:
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
        missing = [a for a in axes if a not in df.columns]
        if missing:
            print(f"WARNING: --axes references columns not in file: {missing}",
                  file=sys.stderr)
            axes = [a for a in axes if a in df.columns]
    else:
        axes = detect_axes(df, exclude={"variant", "hidden"})

    metric = args.primary_metric

    with out_path.open("w", encoding="utf-8") as out:
        # Header / file shape
        out.write(f"COLLATED RESULTS ANALYSIS\n")
        out.write(f"Source file       : {in_path.resolve()}\n")
        out.write(f"Sheet             : {sheet}\n")
        out.write(f"Rows x Cols       : {n_rows} x {n_cols}\n")
        out.write(f"All columns       : {df.columns.tolist()}\n")
        out.write(f"Detected axes     : {axes}\n")
        out.write(f"Primary metric    : {metric}\n")

        # 1. AXIS SUMMARY
        out.write(fmt_section("1. AXIS SUMMARY  (how often each level appears)"))
        if not axes:
            out.write("(no config-axis columns detected; the report will treat "
                      "every row as a separate variant only.)\n")
        for ax in axes:
            vc = df[ax].value_counts(dropna=False).to_dict()
            out.write(f"\n  {ax}: {vc}")
        out.write("\n")

        # Numeric metric schema
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        out.write(fmt_section("2. NUMERIC METRIC COLUMNS"))
        out.write(", ".join(num_cols))
        out.write("\n")

        # Quick takeaways (Claude reads this first when you paste back)
        write_quick_takeaways(out, df, axes, metric)

        cols = display_cols(df, axes)

        # 3. TOP-N OVERALL by each global metric
        out.write(fmt_section("3. TOP-N OVERALL"))
        for m in TOP_METRICS_OVERALL:
            write_top_n(out, df, m, args.top_n, cols)

        # 4. TOP-N per region
        out.write(fmt_section("4. TOP-N PER REGION"))
        for m in REGION_METRICS:
            write_top_n(out, df, m, args.top_n, cols)

        # 5. TOP-N by recall@p
        out.write(fmt_section("5. TOP-N BY RECALL@P  (overall)"))
        for m in RECALL_METRICS:
            write_top_n(out, df, m, args.top_n, cols)

        # 6. TOP-N by per-region recall@p
        out.write(fmt_section("6. TOP-N BY PER-REGION RECALL@P"))
        for m in REGION_RECALL_METRICS:
            write_top_n(out, df, m, args.top_n, cols)

        # 7. BOTTOM-5 sanity
        out.write(fmt_section("7. SANITY: BOTTOM ROWS"))
        write_bottom_n(out, df, metric, 5, cols)

        # 8. Per-axis sensitivity
        write_axis_sensitivity(out, df, axes, metric)

        # 9. 2-axis cross-tabs
        write_2axis_crosstabs(out, df, axes, metric)

        # 10. Variant win counts
        write_variant_wins(out, df, axes, metric, top_n_in_group=5)

        # 11. Variant stability
        write_variant_stability(out, df, metric)

        # 12. Best per (head, test_set)
        write_best_per_head_testset(out, df, cols)

        # 13. Region gap
        write_region_gap(out, df, axes)

        # ---- Recipe-focused sections (added for COMMANDSIMP R0-R5 evaluation) ----
        # These are the sections to paste back; they directly answer
        # "did per-client std / few-shot / augs move the audios6 number?"

        # 14. Delta vs baseline (per (head, variant), per recipe)
        if args.baseline_recipe:
            write_recipe_delta_vs_baseline(
                out, df,
                baseline=args.baseline_recipe,
                metrics=RECIPE_METRICS,
            )

        # 15. Recipe x variant pivot
        write_recipe_variant_pivot(out, df, metric=metric)

        # 16. Best variant per (head, recipe)
        write_best_per_recipe(out, df, axes)

        # 17. Arch family x recipe (linear vs tiny vs default check)
        write_arch_family_summary(out, df)

        # 18. Calibration focus
        write_calibration_focus(out, df)

        # 19. Per-recipe region performance (IND vs PHP)
        write_region_per_recipe(out, df)

        # Closing note
        out.write(fmt_section("END OF REPORT"))
        out.write(f"Written to: {out_path.resolve()}\n")
        out.write(f"Paste this file's contents into chat for analysis.\n")

    print(f"Wrote report -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
