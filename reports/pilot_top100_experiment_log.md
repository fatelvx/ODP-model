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

### Summary CPU Max-Notes Check

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_summary_cpu_real_m7000 `
  --model summary `
  --epochs 12 `
  --patience 4 `
  --batch-size 8 `
  --grad-accum-steps 2 `
  --checkpoint-metric val_mean_mae `
  --max-notes 7000 `
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
| Validation mean MAE | 0.02435 | 0.01616 |
| Validation pairwise order | 45.96% | 52.02% |

Holdout test metrics:

| Target | MAE | R2 | Pairwise | MAE improvement vs train-mean baseline | MAE improvement vs difficulty rating |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.016027 | -0.0040 | 56.41% | 27.41% | 29.57% |
| acc_std | 0.020069 | -0.0608 | 43.59% | 30.26% | 32.62% |
| skill_gap | 0.029296 | -0.0130 | 57.69% | 23.28% | 24.85% |

Notes:

- Raising `max_notes` from 3000 to 7000 did not improve summary-model MAE on
  this holdout; mean MAE moved from 0.02156 to 0.02180.
- Holdout pairwise order improved from 45.30% to 52.56% on average, but the
  validation pairwise curve was much weaker than the m3000 run.
- Keep `pilot_top100_summary_cpu_real_clean` as the summary MAE baseline for
  now. Use higher `MAX_NOTES` primarily for real sequence models on GPU, not as
  a guaranteed summary-model improvement.

### LSTM CPU Feasibility Run

Full-length CPU attempt:

- `pilot_top100_lstm_cpu_real_m7000` used `max_notes=7000`, embed 32/hidden
  64, batch size 4, and did not finish epoch 1 within 10 minutes on CPU.
- The stale partial run directory only contains a history header and should not
  be used for comparison.

Completed truncated CPU command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_lstm_cpu_real_m1200 `
  --model lstm `
  --epochs 3 `
  --patience 2 `
  --batch-size 8 `
  --grad-accum-steps 2 `
  --checkpoint-metric val_mean_mae `
  --max-notes 1200 `
  --group-column beatmapset_id `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --huber-delta 0.5 `
  --device cpu `
  --amp off `
  --lstm-embed-dim 16 `
  --lstm-hidden-dim 32 `
  --lstm-layers 1 `
  --lstm-dropout 0.0 `
  --lstm-head-dropout 0.2
```

Truncation audit for this CPU run:

| max_notes | Rows truncated | Truncation rate | Max notes over limit |
| ---: | ---: | ---: | ---: |
| 1200 | 42 / 93 | 45.16% | 5214 |

Validation curve:

| Metric | Start | Final |
| --- | ---: | ---: |
| Validation mean MAE | 0.02357 | 0.02137 |
| Validation pairwise order | 42.42% | 46.97% |

Runtime:

| Metric | Value |
| --- | ---: |
| Average epoch seconds | 62.04 |

Holdout test metrics:

| Target | MAE | R2 | Pairwise | MAE improvement vs train-mean baseline | MAE improvement vs difficulty rating |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.018690 | -0.0626 | 48.72% | 15.35% | 17.87% |
| acc_std | 0.022163 | -0.1286 | 30.77% | 22.99% | 25.59% |
| skill_gap | 0.036376 | -0.1610 | 42.31% | 4.74% | 6.68% |

Notes:

- This proves the LSTM training/evaluation pipeline works on real pilot data.
- It is not a good quality comparison because 45.16% of maps are truncated.
- CPU LSTM is too slow for full-length pilot training. Use Colab CUDA for the
  next real sequence model run.

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
- This was the best small-data ranking baseline before the pairwise sweep below.

### Tabular Forest Pairwise Sweep

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.sweep_forest `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --out-dir outputs\forest_sweep_pilot_top100_pairwise_real `
  --max-notes 7000 `
  --group-column beatmapset_id `
  --cv-folds 5 `
  --seed 42 `
  --trees 200,500,800 `
  --min-samples-leaf 1,2,4 `
  --max-features sqrt,0.75,1.0 `
  --feature-sets core,burst `
  --selection-metric mean_pairwise_order_accuracy `
  --workers -1
```

Best sweep candidate:

| Candidate | Feature Set | Trees | Leaf | Max Features | Mean MAE | Mean R2 | Mean Spearman | Mean Pairwise | Mean Improvement |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| core_trees200_leaf2_featsqrt | core | 200 | 2 | sqrt | 0.023484 | 0.1166 | 0.4494 | 65.13% | 13.44% |

Top sweep rows by pairwise order:

| Candidate | Mean MAE | Mean R2 | Mean Spearman | Mean Pairwise |
| --- | ---: | ---: | ---: | ---: |
| core_trees200_leaf2_featsqrt | 0.023484 | 0.1166 | 0.4494 | 65.13% |
| core_trees500_leaf2_featsqrt | 0.023816 | 0.1025 | 0.4097 | 63.79% |
| core_trees800_leaf2_featsqrt | 0.023877 | 0.0960 | 0.4020 | 63.64% |
| core_trees200_leaf1_featsqrt | 0.024178 | 0.0891 | 0.3918 | 63.46% |
| core_trees800_leaf2_feat0.75 | 0.024122 | 0.0490 | 0.3941 | 63.44% |

Best-params training command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_forest_core_pairwise_best_real `
  --model tabular_forest `
  --feature-set core `
  --forest-trees 200 `
  --forest-min-samples-leaf 2 `
  --forest-max-features sqrt `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --max-notes 7000 `
  --workers -1 `
  --seed 42
```

Best-params 5-fold grouped out-of-fold metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.017443 | 0.1095 | 0.4729 | 65.97% | 12.28% |
| acc_std | 0.023182 | 0.1331 | 0.4170 | 64.07% | 15.42% |
| skill_gap | 0.029827 | 0.1070 | 0.4584 | 65.36% | 12.63% |

Notes:

- The pairwise sweep improved the small-data forest ranking baseline from
  63.79% to 65.13% mean pairwise order and reduced mean CV MAE from 0.02382 to
  0.02348.
- The best setting is smaller than the previous 500-tree baseline: 200 trees,
  leaf 2, `sqrt` features, core feature set.
- Top feature importances in the best model start with `notes_per_sec`,
  `delta_p90`, `ln_ratio`, `short_gap_50ms_ratio`, and `delta_p50`.
- Fold 5 remains weak and slightly negative-rank, so this is an incremental
  pilot improvement, not final evidence of model quality.

### Tabular Forest Score-Count Weight Check

Code change:

- `tabular_forest` now passes `sample_weight` into sklearn `fit(...)` for both
  holdout training and grouped CV folds when `--sample-weight-column` is set.
- Before this change, forest runs only recorded sample-weight metadata; neural
  runs used the weights, but forests did not.

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_forest_core_pairwise_weighted_real `
  --model tabular_forest `
  --feature-set core `
  --forest-trees 200 `
  --forest-min-samples-leaf 2 `
  --forest-max-features sqrt `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --max-notes 7000 `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --workers -1 `
  --seed 42
```

5-fold grouped out-of-fold metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.017258 | 0.0913 | 0.4557 | 65.24% | 13.21% |
| acc_std | 0.022799 | 0.0968 | 0.3989 | 63.39% | 16.81% |
| skill_gap | 0.029509 | 0.0893 | 0.4443 | 64.75% | 13.56% |

Comparison against unweighted pairwise-best forest:

| Run | Mean CV MAE | Mean CV R2 | Mean CV Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: |
| Unweighted pairwise-best | 0.023484 | 0.1166 | 65.13% | 13.44% |
| Score-count weighted | 0.023189 | 0.0920 | 64.46% | 14.52% |

Notes:

- Score-count weighting improved mean CV MAE and mean baseline improvement, but
  reduced R2 and pairwise ordering.
- Keep `pilot_top100_forest_core_pairwise_best_real` as the ranking baseline.
  Keep `pilot_top100_forest_core_pairwise_weighted_real` as the reliability/MAE
  comparison run.

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
- `outputs\forest_sweep_pilot_top100_pairwise_real\sweep_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_clean\run_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_m7000\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m1200\run_report.html`
- `outputs\runs\pilot_top100_forest_core_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_best_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_weighted_real\run_report.html`
- `outputs\runs\pilot_top100_forest_burst_real\run_report.html`

## Current Decision

Do not trust the summary model as "good" yet. It reduces MAE on one holdout
split, but its ranking signal is weak and predictions are compressed.
The m7000 summary check did not beat the m3000 summary run on MAE, so raising
`MAX_NOTES` is not automatically helpful for the summary model.

Use `pilot_top100_forest_core_pairwise_best_real` as the current small-data
ranking baseline, because the pairwise sweep improved grouped CV ranking and
MAE while using fewer trees than the previous core forest.
Use `pilot_top100_forest_core_pairwise_weighted_real` only when comparing
label-reliability weighting: it improves mean CV MAE but slightly hurts ranking.
Treat `pilot_top100_lstm_cpu_real_m1200` only as a pipeline/performance proof:
the run is heavily truncated and ranking is weaker than the current baselines.

Next training iteration:

1. Prefer `feature-set=core`, 200 trees, leaf 2, and `sqrt` max features for
   tabular pilot comparisons.
2. Increase `MAX_NOTES` above 3000 for real pilot/Colab runs when memory allows,
   because 15.05% of pilot maps exceed 3000 notes.
3. Run Colab/GPU LSTM against the same pilot dataset with `MAX_NOTES >= 3000`
   and preferably near 7000 if memory allows; CPU full-length LSTM did not
   finish epoch 1 within 10 minutes.
4. Add more top100 maps before claiming model quality; 93 maps is still a pilot
   dataset.
