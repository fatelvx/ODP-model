# Pilot Top100 Experiment Log

Date: 2026-06-06

This log records real training runs on the local pilot top100 dataset. Large
artifacts under `outputs/` are intentionally gitignored; this file keeps the
portable summary in git.

## Data Audit

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.audit_dataset `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --max-notes 3000 `
  --out-dir outputs\dataset_audit_pilot_top100_real
```

Results:

| Item | Value |
| --- | ---: |
| Label rows | 93 |
| Usable rows | 93 |
| Missing sequences | 0 |
| Extra sequence files | 7 |
| Beatmapset groups | 27 |
| Full top100 rows | 78 / 93 |
| Full top100 rate | 83.87% |
| Low score-count rows | 12 / 93 |
| Low score-count rate | 12.90% |
| Median sequence length | 1078 notes |
| Max sequence length | 6414 notes |
| Rows exceeding max_notes=3000 | 14 / 93 |
| Truncation rate at max_notes=3000 | 15.05% |

Audit warnings:

- `small_usable_dataset`: use this as a pilot/smoke dataset, not final proof.
- `sequence_truncation`: 15.05% of maps exceed `max_notes=3000`.

## Real Training Runs

### Summary CPU Baseline

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_summary_cpu_real_clean `
  --model summary `
  --epochs 12 `
  --patience 4 `
  --batch-size 8 `
  --grad-accum-steps 2 `
  --checkpoint-metric val_mean_mae `
  --max-notes 3000 `
  --group-column beatmapset_id `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --huber-delta 0.5 `
  --device cpu `
  --amp off
```

Validation curve:

| Metric | Start | Final |
| --- | ---: | ---: |
| Validation mean MAE | 0.02477 | 0.01597 |
| Validation pairwise order | 56.57% | 78.28% |

Holdout test metrics:

| Target | MAE | R2 | Pairwise | MAE improvement vs train-mean baseline | MAE improvement vs difficulty rating |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.015991 | -0.0025 | 48.72% | 27.57% | 29.73% |
| acc_std | 0.019717 | -0.0601 | 33.33% | 31.49% | 33.80% |
| skill_gap | 0.028973 | -0.0116 | 53.85% | 24.13% | 25.68% |

Notes:

- MAE beats both simple baselines on this holdout split.
- R2 is still negative and pairwise ranking is weak, especially for `acc_std`.
- Prediction spread is too compressed, so the model is mostly learning the
  center of the label distribution.

### Tabular Forest Core

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_forest_core_real `
  --model tabular_forest `
  --feature-set core `
  --forest-trees 500 `
  --forest-min-samples-leaf 2 `
  --forest-max-features sqrt `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --max-notes 7000 `
  --workers -1 `
  --seed 42
```

5-fold grouped out-of-fold metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.017690 | 0.0961 | 0.4308 | 64.63% | 11.04% |
| acc_std | 0.023541 | 0.1172 | 0.3751 | 62.74% | 14.11% |
| skill_gap | 0.030216 | 0.0943 | 0.4234 | 64.00% | 11.49% |

Notes:

- Core forest is weaker than summary on the tiny holdout MAE, but CV is much
  healthier: positive R2 for every target and around 64% pairwise ordering.
- This is the best small-data ranking baseline so far.

### Tabular Forest Burst

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_forest_burst_real `
  --model tabular_forest `
  --feature-set burst `
  --forest-trees 500 `
  --forest-min-samples-leaf 2 `
  --forest-max-features sqrt `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --max-notes 7000 `
  --workers -1 `
  --seed 42
```

5-fold grouped out-of-fold metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.018383 | 0.0710 | 0.3555 | 62.06% | 7.55% |
| acc_std | 0.024573 | 0.0876 | 0.3032 | 60.26% | 10.34% |
| skill_gap | 0.031456 | 0.0692 | 0.3435 | 61.52% | 7.85% |

Notes:

- Burst features did not help on this pilot dataset.
- Keep `feature-set=core` as the current small-data tabular baseline.

## Generated Local Reports

- `outputs\pilot_top100_real_dashboard.html`
- `outputs\pilot_top100_real_comparison.html`
- `outputs\pilot_top100_real_comparison.csv`
- `outputs\pilot_top100_real_decision_summary.csv`
- `outputs\runs\pilot_top100_summary_cpu_real_clean\run_report.html`
- `outputs\runs\pilot_top100_forest_core_real\run_report.html`
- `outputs\runs\pilot_top100_forest_burst_real\run_report.html`

## Current Decision

Do not trust the summary model as "good" yet. It reduces MAE on one holdout
split, but its ranking signal is weak and predictions are compressed.

Use `pilot_top100_forest_core_real` as the current small-data ranking baseline,
because the 5-fold grouped CV result is more stable than the tiny 13-map holdout.

Next training iteration:

1. Prefer `feature-set=core` for tabular pilot comparisons.
2. Increase `MAX_NOTES` above 3000 for real pilot/Colab runs when memory allows,
   because 15.05% of pilot maps exceed 3000 notes.
3. Run Colab/GPU LSTM against the same pilot dataset with a larger `MAX_NOTES`
   and compare against forest core CV, not only holdout MAE.
4. Add more top100 maps before claiming model quality; 93 maps is still a pilot
   dataset.
