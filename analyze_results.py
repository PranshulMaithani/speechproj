#!/usr/bin/env python3
"""
Analyze inference results from predict_cpu.py

Usage:
    python analyze_results.py --results cheaters_results.json
    python analyze_results.py --results cheaters_results.json --output analysis.csv
"""

import json
import csv
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics


# ============================================================
# Helpers
# ============================================================

def extract_speaker_id(filepath: str) -> str:
    """Extract speaker ID from path like .../speaker_id/answer.wav"""
    p = Path(filepath)
    parent = p.parent.name
    if parent and parent != p.parent.parent.name:
        return parent
    return p.stem


def compute_file_stats(result: dict) -> dict:
    """Compute detailed statistics for a single file result."""
    windows = result.get("window_predictions", [])
    segments = result.get("segments", [])

    total_windows = len(windows)
    silence_windows = sum(1 for w in windows if w["label"] == "silence")
    read_windows = sum(1 for w in windows if w["label"] == "read")
    spon_windows = sum(1 for w in windows if w["label"] == "spontaneous")

    silence_duration = sum(s["duration_sec"] for s in segments if s["label"] == "silence")
    read_duration = sum(s["duration_sec"] for s in segments if s["label"] == "read")
    spon_duration = sum(s["duration_sec"] for s in segments if s["label"] == "spontaneous")
    speech_duration = read_duration + spon_duration

    read_confs = [w["confidence"] for w in windows if w["label"] == "read"]
    spon_confs = [w["confidence"] for w in windows if w["label"] == "spontaneous"]

    speech_ratios = [w["speech_ratio"] for w in windows]
    avg_speech_ratio = statistics.mean(speech_ratios) if speech_ratios else 0

    avg_read_conf = statistics.mean(read_confs) if read_confs else 0
    min_read_conf = min(read_confs) if read_confs else 0
    avg_spon_conf = statistics.mean(spon_confs) if spon_confs else 0

    speaking_labels = [w["label"] for w in windows if w["label"] != "silence"]
    transitions = sum(1 for i in range(1, len(speaking_labels))
                      if speaking_labels[i] != speaking_labels[i-1])

    non_silence_segs = [s for s in segments if s["label"] != "silence"]
    first_label = non_silence_segs[0]["label"] if non_silence_segs else "none"
    last_label = non_silence_segs[-1]["label"] if non_silence_segs else "none"

    return {
        "filepath": result["filepath"],
        "filename": result["filename"],
        "speaker_id": extract_speaker_id(result["filepath"]),
        "duration_sec": result["duration_sec"],
        "overall_label": result["overall_label"],
        "overall_confidence": result["overall_confidence"],
        "read_ratio": result["read_ratio"],
        "total_windows": total_windows,
        "silence_windows": silence_windows,
        "read_windows": read_windows,
        "spon_windows": spon_windows,
        "speaking_windows": read_windows + spon_windows,
        "silence_pct": silence_windows / max(total_windows, 1),
        "silence_duration": silence_duration,
        "read_duration": read_duration,
        "spon_duration": spon_duration,
        "speech_duration": speech_duration,
        "avg_speech_ratio": avg_speech_ratio,
        "avg_read_conf": avg_read_conf,
        "min_read_conf": min_read_conf,
        "avg_spon_conf": avg_spon_conf,
        "transitions": transitions,
        "first_label": first_label,
        "last_label": last_label,
        "is_mixed": transitions > 0,
    }


# ============================================================
# Speaker-Level Aggregation
# ============================================================

def aggregate_speakers(file_stats: list) -> dict:
    speakers = defaultdict(list)
    for fs in file_stats:
        speakers[fs["speaker_id"]].append(fs)

    speaker_summaries = {}
    for spk_id, files in speakers.items():
        n_read = sum(1 for f in files if f["overall_label"] == "read")
        n_spon = sum(1 for f in files if f["overall_label"] == "spontaneous")
        n_silence = sum(1 for f in files if f["overall_label"] == "silence")
        read_ratios = [f["read_ratio"] for f in files]
        confs = [f["overall_confidence"] for f in files]

        if n_read >= 2:
            risk = "HIGH"
        elif n_read == 1:
            risk = "MEDIUM"
        elif statistics.mean(read_ratios) > 0.3:
            risk = "BORDERLINE"
        else:
            risk = "LOW"

        speaker_summaries[spk_id] = {
            "speaker_id": spk_id,
            "n_files": len(files),
            "n_read": n_read,
            "n_spon": n_spon,
            "n_silence": n_silence,
            "avg_read_ratio": statistics.mean(read_ratios),
            "max_read_ratio": max(read_ratios),
            "avg_confidence": statistics.mean(confs),
            "total_speech_sec": sum(f["speech_duration"] for f in files),
            "total_silence_sec": sum(f["silence_duration"] for f in files),
            "risk": risk,
            "files": files,
        }

    return speaker_summaries


# ============================================================
# Error Analysis
# ============================================================

def analyze_errors(file_stats: list) -> dict:
    buckets = {
        "true_positive": [],
        "missed_cheater": [],
        "silence_files": [],
        "borderline": [],
        "high_silence_fraction": [],
        "very_short_speech": [],
        "mixed_behavior": [],
        "low_confidence": [],
    }

    for fs in file_stats:
        if fs["overall_label"] == "silence":
            buckets["silence_files"].append(fs)
            continue

        if fs["overall_label"] == "read":
            buckets["true_positive"].append(fs)
        else:
            buckets["missed_cheater"].append(fs)

        if 0.30 <= fs["read_ratio"] <= 0.60:
            buckets["borderline"].append(fs)
        if fs["silence_pct"] > 0.50:
            buckets["high_silence_fraction"].append(fs)
        if fs["speech_duration"] < 10.0:
            buckets["very_short_speech"].append(fs)
        if fs["is_mixed"]:
            buckets["mixed_behavior"].append(fs)
        if fs["overall_confidence"] < 0.80:
            buckets["low_confidence"].append(fs)

    return buckets


def find_misclassification_patterns(missed: list) -> dict:
    if not missed:
        return {"count": 0}

    return {
        "count": len(missed),
        "avg_speech_duration": statistics.mean([f["speech_duration"] for f in missed]),
        "avg_silence_pct": statistics.mean([f["silence_pct"] for f in missed]),
        "avg_speech_ratio": statistics.mean([f["avg_speech_ratio"] for f in missed]),
        "avg_spon_confidence": statistics.mean([f["avg_spon_conf"] for f in missed]),
        "avg_read_ratio": statistics.mean([f["read_ratio"] for f in missed]),
        "n_high_silence": sum(1 for f in missed if f["silence_pct"] > 0.50),
        "n_short_speech": sum(1 for f in missed if f["speech_duration"] < 10),
        "n_mixed": sum(1 for f in missed if f["is_mixed"]),
        "n_very_confident_wrong": sum(1 for f in missed if f["overall_confidence"] > 0.95),
        "speakers_affected": list(set(f["speaker_id"] for f in missed)),
    }


# ============================================================
# Report
# ============================================================

def print_header(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_analysis(file_stats, speaker_summaries, buckets, miss_patterns):
    total = len(file_stats)

    print_header("INFERENCE ANALYSIS REPORT")
    print(f"  Total files analyzed: {total}")
    print(f"  Total speakers: {len(speaker_summaries)}")
    print(f"  Model said READ: {len(buckets['true_positive'])}")
    print(f"  Model said SPONTANEOUS: {len(buckets['missed_cheater'])}")
    print(f"  Model said SILENCE: {len(buckets['silence_files'])}")
    print(f"  Detection rate: {len(buckets['true_positive'])/max(total,1):.1%}")

    print_header("SPEAKER-LEVEL SUMMARY", "-")
    risk_order = {"HIGH": 0, "MEDIUM": 1, "BORDERLINE": 2, "LOW": 3}
    sorted_speakers = sorted(speaker_summaries.values(),
                              key=lambda s: (risk_order.get(s["risk"], 4), -s["avg_read_ratio"]))

    print(f"  {'Speaker':<20s} {'Files':>5s} {'Read':>5s} {'Spon':>5s} "
          f"{'AvgReadRatio':>12s} {'Speech(s)':>10s} {'Risk':<10s}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*12} {'-'*10} {'-'*10}")

    for s in sorted_speakers:
        print(f"  {s['speaker_id']:<20s} {s['n_files']:>5d} {s['n_read']:>5d} {s['n_spon']:>5d} "
              f"{s['avg_read_ratio']:>11.1%} {s['total_speech_sec']:>9.1f} {s['risk']:<10s}")

    risk_counts = defaultdict(int)
    for s in speaker_summaries.values():
        risk_counts[s["risk"]] += 1

    print(f"\n  Risk distribution:")
    for risk in ["HIGH", "MEDIUM", "BORDERLINE", "LOW"]:
        count = risk_counts.get(risk, 0)
        bar = "X" * count
        print(f"    {risk:<12s} {count:>3d} speakers  {bar}")

    if buckets["missed_cheater"]:
        print_header("MISSED CHEATERS (Model said SPONTANEOUS)", "-")
        print(f"  Count: {len(buckets['missed_cheater'])} / {total} "
              f"({len(buckets['missed_cheater'])/max(total,1):.1%})")
        print()

        for f in sorted(buckets["missed_cheater"], key=lambda x: x["overall_confidence"], reverse=True):
            print(f"  >> {f['speaker_id']}/{f['filename']}")
            print(f"     Conf: {f['overall_confidence']:.1%} spon | "
                  f"ReadRatio: {f['read_ratio']:.1%} | "
                  f"Speech: {f['speech_duration']:.1f}s | "
                  f"Silence: {f['silence_pct']:.0%} | "
                  f"Mixed: {'Yes' if f['is_mixed'] else 'No'} | "
                  f"SpeechRatio: {f['avg_speech_ratio']:.0%}")

        if miss_patterns.get("count", 0) > 0:
            print(f"\n  --- Patterns in missed cheaters ---")
            print(f"  Avg speech duration: {miss_patterns['avg_speech_duration']:.1f}s")
            print(f"  Avg silence fraction: {miss_patterns['avg_silence_pct']:.1%}")
            print(f"  Avg read ratio: {miss_patterns['avg_read_ratio']:.1%}")
            print(f"  Avg spon confidence: {miss_patterns['avg_spon_confidence']:.1%}")
            print(f"  High silence (>50%): {miss_patterns['n_high_silence']} files")
            print(f"  Short speech (<10s): {miss_patterns['n_short_speech']} files")
            print(f"  Mixed behavior: {miss_patterns['n_mixed']} files")
            print(f"  Very confident wrong (>95%): {miss_patterns['n_very_confident_wrong']} files")
            print(f"  Speakers affected: {len(miss_patterns['speakers_affected'])}")

    if buckets["borderline"]:
        print_header("BORDERLINE CASES (Read ratio 30-60%)", "-")
        for f in sorted(buckets["borderline"], key=lambda x: x["read_ratio"], reverse=True):
            v = "READ" if f["overall_label"] == "read" else "SPON"
            print(f"  {f['speaker_id']}/{f['filename']:<30s} "
                  f"Verdict={v} ReadRatio={f['read_ratio']:.1%} Transitions={f['transitions']}")

    if buckets["high_silence_fraction"]:
        print_header("HIGH SILENCE FILES (>50% silence windows)", "-")
        for f in buckets["high_silence_fraction"]:
            print(f"  {f['speaker_id']}/{f['filename']:<30s} "
                  f"Silence={f['silence_pct']:.0%} Speech={f['speech_duration']:.1f}s")

    if buckets["very_short_speech"]:
        print_header("VERY SHORT SPEECH (<10s actual speech)", "-")
        for f in buckets["very_short_speech"]:
            print(f"  {f['speaker_id']}/{f['filename']:<30s} "
                  f"Speech={f['speech_duration']:.1f}s Duration={f['duration_sec']:.1f}s")

    missed_count = len(buckets["missed_cheater"])
    low_risk = sum(1 for s in speaker_summaries.values() if s["risk"] == "LOW")

    print_header("NEXT STEPS", "=")
    print(f"""
  1. LISTEN to the {missed_count} missed cheater files
     - Are they actually reading? -> model failed, need retraining
     - Are they genuinely spontaneous? -> SME was wrong (had video you dont)

  2. CHECK {low_risk} LOW-risk speakers (model says all spontaneous)
     - If SME was SURE: serious model blind spot on those speakers
     - If SME only SUSPECTED: model might be right

  3. REVIEW {len(buckets['borderline'])} borderline files (read ratio 30-60%)
     - These are the toughest calls - listen and manually label

  4. CHECK {len(buckets['high_silence_fraction'])} high-silence files
     - Candidate barely spoke? Recording issue?
     - Exclude these from retraining data

  5. FILL IN the exported CSVs:
     - analysis_report_files.csv  -> your_label column (read/spontaneous)
     - analysis_report_speakers.csv -> sme_confidence (sure/suspected)
     - This becomes your ground truth for retraining
""")


def export_csv(file_stats, speaker_summaries, output_path):
    file_csv = output_path.replace(".csv", "_files.csv")
    with open(file_csv, "w", newline="", encoding="utf-8") as f:
        cols = [
            "speaker_id", "filename", "filepath",
            "overall_label", "overall_confidence", "read_ratio",
            "duration_sec", "speech_duration", "silence_duration", "silence_pct",
            "read_windows", "spon_windows", "silence_windows", "total_windows",
            "avg_read_conf", "avg_spon_conf", "avg_speech_ratio",
            "transitions", "is_mixed", "first_label", "last_label",
            "your_label", "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for fs in sorted(file_stats, key=lambda x: (x["speaker_id"], x["filename"])):
            row = {k: v for k, v in fs.items() if k in cols}
            row["your_label"] = ""
            row["notes"] = ""
            for c in ["overall_confidence", "read_ratio", "silence_pct",
                       "avg_read_conf", "avg_spon_conf", "avg_speech_ratio"]:
                if c in row and isinstance(row[c], float):
                    row[c] = round(row[c], 3)
            writer.writerow(row)
    print(f"  File-level CSV:    {file_csv}")

    speaker_csv = output_path.replace(".csv", "_speakers.csv")
    with open(speaker_csv, "w", newline="", encoding="utf-8") as f:
        cols = [
            "speaker_id", "n_files", "n_read", "n_spon", "n_silence",
            "avg_read_ratio", "max_read_ratio", "avg_confidence",
            "total_speech_sec", "total_silence_sec", "risk",
            "sme_confidence", "your_verdict", "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for s in sorted(speaker_summaries.values(),
                       key=lambda x: ({"HIGH":0,"MEDIUM":1,"BORDERLINE":2,"LOW":3}.get(x["risk"],4))):
            row = {k: v for k, v in s.items() if k in cols}
            row["sme_confidence"] = ""
            row["your_verdict"] = ""
            row["notes"] = ""
            for c in ["avg_read_ratio", "max_read_ratio", "avg_confidence"]:
                if c in row and isinstance(row[c], float):
                    row[c] = round(row[c], 3)
            writer.writerow(row)
    print(f"  Speaker-level CSV: {speaker_csv}")


def main():
    parser = argparse.ArgumentParser(description="Analyze inference results")
    parser.add_argument("--results", required=True, help="JSON results from predict_cpu.py")
    parser.add_argument("--output", default="analysis_report.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading: {args.results}")
    with open(args.results) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]

    print(f"Loaded {len(raw)} files\n")

    file_stats = [compute_file_stats(r) for r in raw]
    speaker_summaries = aggregate_speakers(file_stats)
    buckets = analyze_errors(file_stats)
    miss_patterns = find_misclassification_patterns(buckets["missed_cheater"])

    print_analysis(file_stats, speaker_summaries, buckets, miss_patterns)

    print_header("EXPORTED FILES", "-")
    export_csv(file_stats, speaker_summaries, args.output)
    print(f"\n  Open CSVs in Excel. Fill your_label + sme_confidence as you listen.\n")


if __name__ == "__main__":
    main()
