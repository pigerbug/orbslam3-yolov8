#!/usr/bin/env python3
"""Summarize Tracking's dynamic_probability_*.txt debug logs."""

import argparse
from pathlib import Path


def data_rows(path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line.split()


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Directory set by DynamicFilter.probabilityDumpPath")
    args = parser.parse_args()
    root = Path(args.log_dir)
    points_file = root / "dynamic_probability_points.txt"
    frames_file = root / "dynamic_probability_frames.txt"
    if not points_file.is_file() or not frames_file.is_file():
        raise SystemExit("Missing dynamic_probability_points.txt or dynamic_probability_frames.txt")

    probabilities = []
    weights = []
    mapping_dynamic = 0
    immune = 0
    frames = set()
    for row in data_rows(points_file):
        # frame timestamp index u v P weight immune mapping_dynamic has_map_point
        if len(row) != 10:
            continue
        frames.add(int(row[0]))
        probabilities.append(float(row[5]))
        weights.append(float(row[6]))
        immune += int(row[7])
        mapping_dynamic += int(row[8])

    frame_rows = list(data_rows(frames_file))
    total = len(probabilities)
    nonzero = sum(value > 0.0 for value in probabilities)
    print(f"logged_frames: {len(frames)}")
    print(f"frame_summary_rows: {len(frame_rows)}")
    print(f"features: {total}")
    if not total:
        return
    print(f"nonzero_probability: {nonzero} ({100.0 * nonzero / total:.2f}%)")
    print(f"mapping_dynamic: {mapping_dynamic} ({100.0 * mapping_dynamic / total:.2f}%)")
    print(f"manhattan_immune: {immune} ({100.0 * immune / total:.2f}%)")
    print(f"Pdynamic mean/p50/p95: {sum(probabilities) / total:.4f} / "
          f"{percentile(probabilities, 0.50):.4f} / {percentile(probabilities, 0.95):.4f}")
    print(f"weight mean/p05/p50: {sum(weights) / total:.4f} / "
          f"{percentile(weights, 0.05):.4f} / {percentile(weights, 0.50):.4f}")


if __name__ == "__main__":
    main()
