#!/usr/bin/env python3
"""Extract tidy, gnuplot-ready .dat files from logs/runs.csv.

Reads logs/runs.csv (the aggregate improvement-progress file written by
log_run.py) and writes two tab-separated files that plot_runs.gp consumes:

  logs/runs_plot.dat   per-config rows:  tag  mode  exact  near  off  mean  metric
  logs/sweep_plot.dat  tightness points: delta  exact  near  off  mean

Offline echo-backend sanity runs (--dry-run) are excluded. Rerun this after
logging new runs (log_run.py does it automatically), then:

  gnuplot logs/plot_runs.gp
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(HERE, "runs.csv")
RUNS_DAT = os.path.join(HERE, "runs_plot.dat")
SWEEP_DAT = os.path.join(HERE, "sweep_plot.dat")


def main():
    rows = []
    with open(RUNS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Offline echo-backend sanity check, not an evaluation run.
            if r.get("tag") == "smoke_dry":
                continue
            if r.get("backend") != "mercury":
                continue
            rows.append(r)

    # Per-config rows: any run with a graded summary (exact column filled).
    config_rows = [r for r in rows if r.get("exact") not in ("", None)]
    with open(RUNS_DAT, "w", encoding="utf-8") as f:
        f.write("# tag\tmode\texact\tnear\toff\tmean\tmetric\n")
        for r in sorted(config_rows, key=lambda r: r.get("tag", "")):
            mean = f"{float(r['mean_score']):.3f}" if r.get("mean_score") else ""
            # Display tags with dashes instead of underscores so they read
            # unambiguously on the chart at any size/font (underscores are
            # easily confused with gnuplot subscripts or clipped by OCR).
            tag = r.get("tag", "?").replace("_", "-")
            f.write("\t".join([
                tag, r.get("mode") or "",
                r.get("exact", ""), r.get("near_miss", ""), r.get("off_target", ""),
                mean, r.get("metric") or "",
            ]) + "\n")

    # Tightness-sweep points: the sweep row carries
    # "delta:exact/near/off/mean;delta:..." in its `deltas` column.
    with open(SWEEP_DAT, "w", encoding="utf-8") as f:
        f.write("# delta\texact\tnear\toff\tmean\n")
        for r in rows:
            if not r.get("deltas"):
                continue
            for chunk in r["deltas"].split(";"):
                d, _, rest = chunk.partition(":")
                parts = rest.split("/")
                if len(parts) == 4:
                    f.write(f"{d}\t{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\n")

    print(f"wrote {RUNS_DAT} ({len(config_rows)} config rows) and {SWEEP_DAT}")


if __name__ == "__main__":
    main()
