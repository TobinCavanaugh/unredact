# Evaluation logs

Every logged run lives here as three artifacts:

| file | contents |
|------|----------|
| `<ts>_<tag>.txt` | full text log (stdout + stderr) of one unredact.py run |
| `<ts>_<tag>.json` | structured sidecar: per-redaction rows + summary (or per-delta rows for `--tightness-sweep`) |
| `runs.csv` | one aggregate row per run — the graphable "improvement progress" file |

## How to log a run

    python log_run.py <tag> [unredact.py args...]

Examples:

    # baseline (semantic scorer, no keyness prior)
    python log_run.py baseline --redactions redactions_broad.json --backend mercury --semantic

    # keyness prior on (current best config)
    python log_run.py prior05 --redactions redactions_broad.json --backend mercury --semantic --prior-weight 0.5

    # echo guard disabled (ablation)
    python log_run.py echo_off --redactions redactions_broad.json --backend mercury --semantic --echo-penalty 0

    # chat mode instead of FIM
    python log_run.py chat --redactions redactions_broad.json --backend mercury --mode chat --semantic

    # isolated Mercury FIM budget ablation (no chat fallback)
    python log_run.py fim_tight --redactions data/synthetic_v2/test.json \
      --backend mercury --mode fim --candidates 6 --full-pool --fim-only

    # constraint-tightness sweep (one candidate pool, re-verified at each delta)
    python log_run.py sweep --redactions redactions_broad.json --backend mercury --semantic --prior-weight 0.5 --tightness-sweep 0,3,5,10

    # ground-truth-free observed-support difficulty diagnostic
    python log_run.py n_eff --redactions redactions_broad.json --backend mercury --n-eff --candidates 8

No API key? Add `--dry-run` to exercise the whole pipeline offline (echo backend).

If `runs.csv` ever gets out of sync with the sidecars (e.g. after a schema
change), rebuild it from the JSON sidecars (the source of truth):

    python log_run.py --rebuild

## runs.csv columns

`ts`, `tag`, `exit`, `redactions_file`, `backend`, `mode`, `candidates`,
`semantic`, `semantic_threshold`, `echo_penalty`, `prior_weight`, `full_pool`,
`loosen`, `tightness_sweep`, `exact`, `near_miss`, `off_target`, `mean_score`,
`metric`, `blind_exact`, `candidate_recall_at_k`, `blind_exact_rate`,
`candidate_recall_rate`, `n_eff_enabled`, `n_eff_mean`, `n_eff_median`, `n_eff_nonempty`,
`n_eff_mean_all_candidates`, `n_eff_mean_in_range_candidates`,
`n_eff_mean_in_range_unique`, `truncated`, `fim_truncated`, `chat_truncated`,
`empty_responses`, `artifact_filtered`, `duplicate_candidates`, `fallback_chat`,
`fim_budget_min`,
`fim_budget_max`, `fim_budget_mean`, `deltas`, `notes`

- `blind_exact` and `candidate_recall_at_k` are the honest deployment metrics.
- `n_eff_*` is a ground-truth-free observed-support proxy. It is computed from
  repeated cleaned draws when the backend provides them; `n_eff = exp(H)` uses
  the empirical frequency distribution, not model probabilities. Empty
  in-range pools are retained per redaction but excluded from mean/median.
- `metric` describes the separate oracle diagnostic score; it must not be read
  as blind performance.
- Sweep runs put a `delta:exact/near/off/mean` list in `deltas` instead of the
  single exact/near/off columns. Tightness sweeps are oracle diagnostics unless
  and until a blind sweep metric is added.
- Current regular runs use sidecar schema `unredact-run/v3`. Their
  `summary.generation` and each row's `generation` block report API attempts,
  max-token truncations, empty responses, FIM artifacts filtered, fallback-chat
  use, and usable candidates. Responses with `finish_reason="length"` are
  discarded and never count toward candidate recall.
