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

### Summary CPU Pairwise Sweep

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.sweep_neural `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --out-dir outputs\neural_sweep_pilot_top100_summary_pairwise_real `
  --run-prefix pilot_top100_summary_pairwise_sweep_real `
  --models summary `
  --epochs 16 `
  --patience 5 `
  --batch-size 8,16 `
  --lrs 0.001,0.0005 `
  --weight-decays 0.0001 `
  --summary-hidden-dims 64,96,128 `
  --summary-dropouts 0.1,0.25 `
  --huber-deltas 0.5 `
  --selection-metric mean_pairwise_order_accuracy `
  --checkpoint-metric val_mean_mae `
  --max-notes 3000 `
  --group-column beatmapset_id `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --grad-accum-steps 2 `
  --grad-clip-norm 1.0 `
  --device cpu `
  --amp off `
  --workers -1 `
  --seed 42
```

Sweep result:

| Item | Value |
| --- | ---: |
| Candidate count | 24 |
| Best candidate | `summary_h128_do0p1_lr0p001_wd0p0001_bs8_hd0p5` |
| Hidden dim | 128 |
| Dropout | 0.1 |
| LR | 0.001 |
| Batch size | 8 |
| Effective batch size | 16 |
| Best epoch | 16 |
| Best validation mean MAE | 0.013813 |
| Best validation pairwise order | 78.79% |

Best-candidate holdout metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.015129 | -0.0019 | 0.0604 | 51.28% | 31.48% |
| acc_std | 0.017679 | -0.0027 | 0.1209 | 52.56% | 38.57% |
| skill_gap | 0.026809 | 0.0047 | 0.0330 | 51.28% | 29.79% |

Comparison against previous summary runs:

| Run | Mean holdout MAE | Mean holdout R2 | Mean holdout Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: |
| Summary m3000 baseline | 0.021560 | -0.0247 | 45.30% | 27.72% |
| Summary m7000 check | 0.021798 | -0.0252 | 52.56% | 26.90% |
| Summary pairwise sweep best | 0.019872 | 0.0000 | 51.71% | 33.28% |

Notes:

- The sweep improved the summary model's holdout MAE and moved average R2 to
  roughly break-even, while also improving holdout pairwise order over the
  original m3000 summary baseline.
- The sweep did not beat the m7000 summary check on holdout pairwise order, and
  validation pairwise was much stronger than holdout pairwise. Treat this as a
  small-data tuning win, not a stable ranking breakthrough.
- Keep `pilot_top100_forest_core_pairwise_best_real` as the stronger ranking
  baseline because it has 5-fold grouped out-of-fold pairwise order near 65%.

### Summary CPU Seed Stability Check

Command shape:

```powershell
$seeds = 7,13,42,99,123
foreach ($seed in $seeds) {
  .\.venv\Scripts\python.exe -m mania_difficulty.train `
    --labels data\processed\labels_pilot_top100.csv `
    --sequences data\processed\sequences_pilot `
    --run-name "pilot_top100_summary_stability_seed${seed}_real" `
    --model summary `
    --epochs 16 `
    --patience 5 `
    --batch-size 8 `
    --grad-accum-steps 2 `
    --checkpoint-metric val_mean_mae `
    --max-notes 3000 `
    --group-column beatmapset_id `
    --sample-weight-column score_count `
    --sample-weight-min 0.25 `
    --sample-weight-max-value 100 `
    --huber-delta 0.5 `
    --lr 0.001 `
    --weight-decay 0.0001 `
    --summary-hidden-dim 128 `
    --summary-dropout 0.1 `
    --device cpu `
    --amp off `
    --workers -1 `
    --seed $seed
}
```

5-seed holdout stability:

| Seed | Mean MAE | Mean R2 | Mean Pairwise | Mean Improvement | Targets beating baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.034483 | -0.6818 | 44.12% | -8.30% | 0 / 3 |
| 13 | 0.017388 | -3.6261 | 66.67% | 32.66% | 3 / 3 |
| 42 | 0.019872 | 0.0000 | 51.71% | 33.28% | 3 / 3 |
| 99 | 0.034943 | -1.3458 | 50.00% | -12.72% | 0 / 3 |
| 123 | 0.019528 | -0.1015 | 45.32% | 31.03% | 3 / 3 |
| Mean +/- std | 0.025243 +/- 0.007780 | -1.1510 +/- 1.3274 | 51.56% +/- 8.06% | 15.19% +/- 21.04% | - |

Validation stability signals:

| Seed | Best validation MAE | Best validation pairwise | Epochs completed | Stop reason |
| ---: | ---: | ---: | ---: | --- |
| 7 | 0.020968 | 61.11% | 11 | early_stopping |
| 13 | 0.032524 | 88.89% | 7 | early_stopping |
| 42 | 0.013813 | 78.79% | 16 | completed |
| 99 | 0.014738 | 88.89% | 6 | early_stopping |
| 123 | 0.028251 | 54.07% | 12 | early_stopping |

Notes:

- This stability check contradicts the optimistic single-split summary result:
  the same parameters swing from useful MAE to worse-than-baseline across
  different group splits.
- Validation pairwise can look excellent while holdout MAE/R2 collapses, so
  single-split validation curves are not enough evidence for model quality on
  93 maps.
- The summary tuning is still useful as a local smoke/performance baseline, but
  it should not drive final model choices without grouped CV, more labels, or
  human judgments.

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

### Tabular Forest Seed Stability Check

Command shape:

```powershell
$seeds = 7,13,42,99,123
foreach ($seed in $seeds) {
  .\.venv\Scripts\python.exe -m mania_difficulty.train `
    --labels data\processed\labels_pilot_top100.csv `
    --sequences data\processed\sequences_pilot `
    --run-name "pilot_top100_forest_core_pairwise_stability_seed${seed}_real" `
    --model tabular_forest `
    --feature-set core `
    --forest-trees 200 `
    --forest-min-samples-leaf 2 `
    --forest-max-features sqrt `
    --cv-folds 5 `
    --group-column beatmapset_id `
    --max-notes 7000 `
    --workers -1 `
    --seed $seed
}
```

5-seed grouped out-of-fold stability:

| Seed | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement | Targets beating baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.026470 | -0.0522 | 0.1911 | 56.05% | 3.99% | 3 / 3 |
| 13 | 0.024748 | 0.0708 | 0.3466 | 61.17% | 9.64% | 3 / 3 |
| 42 | 0.023484 | 0.1166 | 0.4494 | 65.13% | 13.44% | 3 / 3 |
| 99 | 0.025798 | -0.0327 | 0.1622 | 55.35% | 6.17% | 3 / 3 |
| 123 | 0.026146 | -0.0585 | 0.1947 | 56.04% | 6.76% | 3 / 3 |
| Mean +/- std | 0.025329 +/- 0.001089 | 0.0088 +/- 0.0713 | 0.2688 +/- 0.1110 | 58.75% +/- 3.82% | 8.00% +/- 3.26% | - |

Holdout sanity check:

| Seed | Mean holdout MAE | Mean holdout R2 | Mean holdout Pairwise | Mean Improvement | Targets beating baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.033949 | -0.3811 | 47.06% | -7.17% | 0 / 3 |
| 13 | 0.016053 | -3.4090 | 55.87% | 38.35% | 3 / 3 |
| 42 | 0.029817 | -0.5234 | 50.43% | -0.57% | 2 / 3 |
| 99 | 0.031920 | -0.9567 | 45.24% | -3.70% | 1 / 3 |
| 123 | 0.025835 | -0.6590 | 59.48% | 8.04% | 3 / 3 |

Notes:

- The forest is more stable than the summary model, especially on grouped
  out-of-fold MAE: the 5-seed CV MAE std is 0.00109 versus the summary
  holdout seed-stability MAE std of 0.00778.
- The seed 42 forest result is optimistic for ranking. The conservative
  multi-seed CV pairwise estimate is about 58.75%, not the single-seed 65.13%.
- Holdout splits remain noisy and sometimes worse than baseline. Use grouped
  out-of-fold metrics for pilot forest decisions until there are more labels.

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
- `outputs\neural_sweep_pilot_top100_summary_pairwise_real\neural_sweep_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_clean\run_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_m7000\run_report.html`
- `outputs\runs\pilot_top100_summary_pairwise_sweep_real_summary_h128_do0p1_lr0p001_wd0p0001_bs8_hd0p5\run_report.html`
- `outputs\runs\pilot_top100_summary_stability_seed{7,13,42,99,123}_real\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m1200\run_report.html`
- `outputs\runs\pilot_top100_forest_core_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_best_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_stability_seed{7,13,42,99,123}_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_weighted_real\run_report.html`
- `outputs\runs\pilot_top100_forest_burst_real\run_report.html`

## Current Decision

Do not trust the summary model as "good" yet. The summary pairwise sweep
improved the holdout mean MAE to 0.01987 and moved holdout pairwise order to
51.71%, but validation ranking was much stronger than holdout ranking, so the
small split is unstable. The 5-seed stability check confirmed this instability:
mean holdout MAE was 0.02524 +/- 0.00778 and two seeds failed all three
train-mean baselines.
The m7000 summary check did not beat the m3000 summary run on MAE, so raising
`MAX_NOTES` is not automatically helpful for the summary model.

Use the core 200-tree forest as the current small-data ranking baseline, but
read the seed 42 run as optimistic. The 5-seed grouped-CV estimate is
0.02533 +/- 0.00109 mean MAE and 58.75% +/- 3.82% pairwise order. This is still
more stable than summary holdout tuning, but not strong enough to claim final
model quality.
Use `pilot_top100_forest_core_pairwise_weighted_real` only when comparing
label-reliability weighting: it improves mean CV MAE but slightly hurts ranking.
Treat `pilot_top100_lstm_cpu_real_m1200` only as a pipeline/performance proof:
the run is heavily truncated and ranking is weaker than the current baselines.

Next training iteration:

1. Prefer `feature-set=core`, 200 trees, leaf 2, and `sqrt` max features for
   tabular pilot comparisons, but report multi-seed CV averages instead of a
   single seed 42 score.
2. Treat `summary_hidden_dim=128`, `summary_dropout=0.1`, `lr=0.001`, and
   effective batch size 16 as a local smoke/tuning baseline only; require
   multi-seed or grouped-CV evidence before trusting summary quality.
3. Increase `MAX_NOTES` above 3000 for real pilot/Colab runs when memory allows,
   because 15.05% of pilot maps exceed 3000 notes.
4. Run Colab/GPU LSTM against the same pilot dataset with `MAX_NOTES >= 3000`
   and preferably near 7000 if memory allows; CPU full-length LSTM did not
   finish epoch 1 within 10 minutes.
5. Add more top100 maps before claiming model quality; 93 maps is still a pilot
   dataset.
