#!/usr/bin/env python3
"""Log an unredact.py evaluation run into logs/ as full text + structured sidecar.

Creates (or reuses) a logs/ directory next to this script and, per run, writes:

  logs/<UTC-timestamp>_<tag>.txt   full text log (stdout + stderr, streamed)
  logs/<UTC-timestamp>_<tag>.json  structured sidecar (written by unredact.py
                                   --json-results: per-redaction rows + summary,
                                   or per-delta rows for --tightness-sweep)
  logs/runs.csv                    one aggregate row per run -- the graphable
                                   "improvement progress" file (spreadsheet /
                                   plotting friendly)

Usage:
  python log_run.py <tag> [unredact.py args...]

Examples:
  python log_run.py baseline --redactions redactions_broad.json --backend mercury --semantic
  python log_run.py prior05 --redactions redactions_broad.json --backend mercury --semantic --prior-weight 0.5
  python log_run.py echo_off --redactions redactions_broad.json --backend mercury --semantic --echo-penalty 0
  python log_run.py chat --redactions redactions_broad.json --backend mercury --mode chat --semantic
  python log_run.py sweep --redactions redactions_broad.json --backend mercury --semantic --prior-weight 0.5 --tightness-sweep 0,3,5,10

The wrapper injects --json-results <sidecar path> itself if you don't pass one.
No API key yet? Add --dry-run to exercise the whole pipeline offline (echo backend).

If runs.csv ever gets out of sync with the sidecars (e.g. after a schema
change), rebuild it from the JSON sidecars (the source of truth):

    python log_run.py --rebuild
"""

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
RUNS_CSV = os.path.join(LOGS_DIR, "runs.csv")

CSV_FIELDS = [
    "ts", "tag", "exit", "redactions_file", "backend", "mode", "candidates",
    "semantic", "semantic_threshold", "echo_penalty", "prior_weight",
    "full_pool", "loosen", "tightness_sweep", "no_length_hint", "fim_only",
    "exact", "near_miss", "off_target", "mean_score", "metric",
    "blind_exact", "candidate_recall_at_k", "blind_exact_rate",
    "candidate_recall_rate", "n_eff_enabled", "n_eff_mean", "n_eff_median", "n_eff_nonempty",
    "n_eff_mean_all_candidates", "n_eff_mean_in_range_candidates",
    "n_eff_mean_in_range_unique", "llada_url", "llada_timeout", "llada_steps",
    "llada_temperature", "llada_remasking", "llada_gen_length", "llada_block_length",
    "truncated", "fim_truncated", "chat_truncated",
    "empty_responses", "artifact_filtered", "fallback_chat", "fim_budget_min",
    "fim_budget_max", "fim_budget_mean", "deltas", "notes",
]


def _row_from_sidecar(ts, tag, data, rc=0, notes=""):
    """Build one runs.csv row from a structured sidecar (source of truth)."""
    row = {"ts": ts, "tag": tag, "exit": rc, "notes": notes}
    m = data.get("meta", {})
    for k in CSV_FIELDS:
        if k in m and k not in row:
            row[k] = m[k]
    if "summary" in data:
        s = data["summary"]
        # Keep legacy plot columns populated from the explicitly labeled oracle
        # diagnostic so old charts continue to work; the honest blind metrics
        # live in their own columns and are the ones to use for performance.
        row.update({
            "exact": s.get("exact", s.get("oracle_exact")),
            "near_miss": s.get("near_miss", s.get("oracle_near_miss")),
            "off_target": s.get("off_target", s.get("oracle_off_target")),
            "mean_score": s.get("mean_score", s.get("oracle_mean_score")),
        })
        row["metric"] = s.get("oracle_metric_label") or s.get("metric_label")
        for k in ("blind_exact", "candidate_recall_at_k", "blind_exact_rate",
                  "candidate_recall_rate"):
            row[k] = s.get(k)
        n_eff = s.get("n_eff") or {}
        row["n_eff_enabled"] = bool(m.get("n_eff", False))
        row.update({
            "n_eff_mean": n_eff.get("mean"),
            "n_eff_median": n_eff.get("median"),
            "n_eff_nonempty": n_eff.get("nonempty_in_range"),
            "n_eff_mean_all_candidates": n_eff.get("mean_all_candidates"),
            "n_eff_mean_in_range_candidates": n_eff.get("mean_in_range_candidates"),
            "n_eff_mean_in_range_unique": n_eff.get("mean_in_range_unique"),
        })
        generation = s.get("generation", {})
        for k in ("truncated", "fim_truncated", "chat_truncated",
                  "empty_responses", "artifact_filtered", "fallback_chat"):
            row[k] = generation.get(k)
        row["fim_budget_min"] = generation.get("fim_requested_budget_min")
        row["fim_budget_max"] = generation.get("fim_requested_budget_max")
        row["fim_budget_mean"] = generation.get("fim_requested_budget_mean")
    if "deltas" in data:
        row["deltas"] = ";".join(
            f"{d['delta']}:{d['exact']}/{d['near_miss']}/{d['off_target']}/"
            f"{('%.3f' % d['mean_score']) if d['mean_score'] is not None else 'n/a'}"
            for d in data["deltas"])
    return row


def _append_csv(row):
    new_file = not os.path.exists(RUNS_CSV)
    if not new_file:
        # A pre-v2 runs.csv may have an older header. Rebuild from sidecars
        # before appending so DictWriter cannot silently create misaligned rows.
        with open(RUNS_CSV, "r", newline="", encoding="utf-8") as f:
            existing_fields = next(csv.reader(f), [])
        if existing_fields != CSV_FIELDS:
            print("warning: runs.csv schema drift detected; rebuilding from sidecars",
                  file=sys.stderr)
            # The current run's sidecar may already be present. Rebuild first,
            # then append only if this row was not recoverable from a sidecar
            # (for example, a run that crashed before writing one).
            rebuild_csv()
            with open(RUNS_CSV, "r", newline="", encoding="utf-8") as f:
                existing_rows = csv.DictReader(f)
                if any(r.get("ts") == str(row.get("ts")) and
                       r.get("tag") == str(row.get("tag"))
                       for r in existing_rows):
                    return

    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _refresh_plot_data():
    """Regenerate the .dat files plot_runs.gp consumes (best-effort)."""
    gen = os.path.join(LOGS_DIR, "make_plot_data.py")
    if os.path.exists(gen):
        try:
            subprocess.run([sys.executable, gen], check=True,
                           capture_output=True, text=True, timeout=60)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"warning: could not refresh plot data: {e}", file=sys.stderr)


def rebuild_csv():
    """Regenerate runs.csv from every sidecar in logs/ (fixes schema drift)."""
    rows = []
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith(".json"):
            continue
        ts = fname.split("_", 1)[0]
        tag = fname[len(ts) + 1:-5]
        with open(os.path.join(LOGS_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        rows.append(_row_from_sidecar(ts, tag, data))
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rebuilt {RUNS_CSV} with {len(rows)} row(s) from sidecars")


def _configure_unicode_stdio():
    """Use UTF-8 for captured logs while tolerating legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    _configure_unicode_stdio()
    if sys.argv[1:] and sys.argv[1] in ("-r", "--rebuild"):
        rebuild_csv()
        return 0
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) == 1 else 1

    timeout = 3600  # seconds; hang-kill safety net (real runs take minutes)
    if "--timeout" in sys.argv:
        i = sys.argv.index("--timeout")
        try:
            timeout = int(sys.argv[i + 1])
        except (ValueError, IndexError):
            print("error: --timeout needs a seconds int", file=sys.stderr)
            return 1
        del sys.argv[i:i + 2]

    tag = sys.argv[1]
    cmd = sys.argv[2:]
    if not cmd:
        print("error: give unredact.py args after the tag", file=sys.stderr)
        return 1

    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{ts}_{tag}"
    txt_path = os.path.join(LOGS_DIR, base + ".txt")
    json_path = os.path.join(LOGS_DIR, base + ".json")

    # Make sure the run writes its structured sidecar where we expect it.
    if "--json-results" in cmd:
        i = cmd.index("--json-results")
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            cmd[i + 1] = json_path
        else:
            cmd.insert(i + 1, json_path)
    else:
        cmd += ["--json-results", json_path]

    full_cmd = [sys.executable, "unredact.py"] + cmd
    child_env = os.environ.copy()
    # The child writes through a pipe on Windows; force UTF-8 rather than the
    # machine's CP1252 locale so raw LLaDA Unicode cannot crash the run.
    child_env["PYTHONIOENCODING"] = "utf-8"
    print(f"==> {ts}  tag={tag}")
    print(f"==> {base}.txt   (full text log)")
    print(f"==> {base}.json  (structured sidecar)")
    print(f"==> running: {' '.join(full_cmd)}")

    with open(txt_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                env=child_env)
        for line in proc.stdout:
            logf.write(line)
            print(line, end="", flush=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
            print(f"==> timed out after {timeout}s; killed", file=sys.stderr)

    row = {"ts": ts, "tag": tag, "exit": rc, "notes": ""}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            row = _row_from_sidecar(ts, tag, data, rc)
        except Exception as e:
            row["notes"] = f"sidecar parse failed: {e}"
    else:
        row["notes"] = "no sidecar written (run may have crashed before grading)"
    if rc != 0 and not row["notes"]:
        row["notes"] = f"exit code {rc}"

    _append_csv(row)
    _refresh_plot_data()
    print(f"==> done (rc={rc}); row appended to logs/runs.csv")
    return rc


if __name__ == "__main__":
    sys.exit(main())
