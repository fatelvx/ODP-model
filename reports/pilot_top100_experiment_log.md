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

### LSTM CPU m3000 Tiny Calibration Run

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_lstm_cpu_real_m3000_tiny_e3 `
  --model lstm `
  --epochs 3 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --patience 3 `
  --checkpoint-metric val_mean_mae `
  --group-column beatmapset_id `
  --max-notes 3000 `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --huber-delta 0.5 `
  --device cpu `
  --amp off `
  --loader-workers 0 `
  --lstm-embed-dim 8 `
  --lstm-hidden-dim 16 `
  --lstm-layers 1 `
  --lstm-dropout 0.0 `
  --lstm-head-dropout 0.2 `
  --seed 42
```

Truncation audit for this CPU run:

| max_notes | Rows truncated | Truncation rate | Max notes over limit |
| ---: | ---: | ---: | ---: |
| 3000 | 14 / 93 | 15.05% | 3414 |

Validation curve:

| Metric | Start | Final |
| --- | ---: | ---: |
| Validation mean MAE | 0.02094 | 0.01994 |
| Validation pairwise order | 50.00% | 55.05% |

Runtime:

| Metric | Value |
| --- | ---: |
| Average epoch seconds | 29.29 |
| Epochs completed | 3 / 3 |
| Effective batch size | 16 |

Holdout test metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline | MAE improvement vs difficulty rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.019138 | -0.0746 | -0.0769 | 46.15% | 13.32% | 15.90% |
| acc_std | 0.019790 | -0.0435 | 0.4615 | 66.67% | 31.23% | 33.55% |
| skill_gap | 0.034360 | -0.1082 | -0.2363 | 42.31% | 10.02% | 11.86% |

Comparison against the previous m1200 LSTM and seed-42 forest holdout:

| Run | Mean holdout MAE | Mean R2 | Mean Spearman | Mean Pairwise | Targets beating train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM m1200 CPU | 0.025743 | -0.1174 | -0.3040 | 40.60% | 3 / 3 |
| LSTM m3000 tiny CPU | 0.024430 | -0.0754 | 0.0495 | 51.71% | 3 / 3 |
| Weighted forest seed 42 | 0.029944 | -0.5343 | -0.0092 | 48.72% | 2 / 3 |

Prediction spread check:

| Target | Actual std | Predicted std | Bias | Over-prediction rate |
| --- | ---: | ---: | ---: | ---: |
| mean_acc | 0.024237 | 0.000056 | -0.006503 | 15.38% |
| acc_std | 0.026729 | 0.000089 | 0.005949 | 84.62% |
| skill_gap | 0.041247 | 0.000114 | 0.013260 | 84.62% |

Notes:

- Raising `max_notes` from 1200 to 3000 and shrinking the model made a CPU
  sequence run feasible: three epochs completed in about 88 seconds of epoch
  time.
- The run improved over the m1200 CPU LSTM on holdout MAE, R2, Spearman, and
  pairwise order, and beat train-mean/difficulty-rating baselines on all three
  target MAEs.
- The predicted standard deviations are still near zero, so the model mostly
  predicts a narrow average band. This is not ready to promote; it is a useful
  Colab parameter calibration run for a longer GPU sequence model.

### LSTM CPU m3000 Tiny 12-Epoch Check

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_lstm_cpu_real_m3000_tiny_e12 `
  --model lstm `
  --epochs 12 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --patience 5 `
  --checkpoint-metric val_mean_mae `
  --group-column beatmapset_id `
  --max-notes 3000 `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --huber-delta 0.5 `
  --device cpu `
  --amp off `
  --loader-workers 0 `
  --lstm-embed-dim 8 `
  --lstm-hidden-dim 16 `
  --lstm-layers 1 `
  --lstm-dropout 0.0 `
  --lstm-head-dropout 0.2 `
  --seed 42
```

Validation curve:

| Metric | Start | Final |
| --- | ---: | ---: |
| Validation mean MAE | 0.02094 | 0.01486 |
| Validation pairwise order | 50.00% | 64.14% |

Runtime:

| Metric | Value |
| --- | ---: |
| Average epoch seconds | 33.70 |
| Total epoch seconds | 404.43 |
| Epochs completed | 12 / 12 |
| Effective batch size | 16 |

Holdout test metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline | MAE improvement vs difficulty rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.015153 | -0.0055 | -0.5275 | 32.05% | 31.37% | 33.41% |
| acc_std | 0.016939 | 0.0007 | 0.3901 | 64.10% | 41.14% | 43.13% |
| skill_gap | 0.027285 | -0.0151 | -0.3132 | 38.46% | 28.55% | 30.01% |

Comparison against earlier sequence CPU checks:

| Run | Mean holdout MAE | Mean R2 | Mean Spearman | Mean Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM m1200 CPU | 0.025743 | -0.1174 | -0.3040 | 40.60% | 14.36% |
| LSTM m3000 tiny e3 | 0.024430 | -0.0754 | 0.0495 | 51.71% | 18.19% |
| LSTM m3000 tiny e12 | 0.019792 | -0.0066 | -0.1502 | 44.87% | 33.69% |

Prediction spread check:

| Target | e3 predicted std | e12 predicted std | Actual std | e12 predicted / actual std | e3 MAE | e12 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.000056 | 0.000080 | 0.024237 | 0.0033 | 0.019138 | 0.015153 |
| acc_std | 0.000089 | 0.000104 | 0.026729 | 0.0039 | 0.019790 | 0.016939 |
| skill_gap | 0.000114 | 0.000233 | 0.041247 | 0.0056 | 0.034360 | 0.027285 |

Notes:

- Longer tiny-LSTM training materially improved validation MAE and holdout MAE,
  and all three targets beat both train-mean and difficulty-rating baselines on
  holdout MAE.
- This did not solve ranking quality. Holdout mean pairwise order fell from
  the e3 run's 51.71% to 44.87%, mostly because `mean_acc` and `skill_gap`
  rank direction are still poor.
- Prediction spread remains the blocker: even after 12 epochs, predicted
  standard deviations are only about 0.33%-0.56% of actual target standard
  deviations. This is still a near-constant predictor with better calibration,
  not a reliable difficulty-ranker.
- For Colab/GPU, do not just extend this exact tiny model forever. Use the e12
  result as evidence that longer sequence training helps MAE, then test a
  larger LSTM and/or stronger ranking objective once GPU runtime is available.

### LSTM CPU m3000 Tiny 12-Epoch Ranking Checkpoint

This run repeats the 12-epoch tiny LSTM but selects `best_model.pt` by
validation pairwise order instead of validation MAE. It tests whether the weak
holdout ranking in the previous run was mostly a checkpoint-selection problem.

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_lstm_cpu_real_m3000_tiny_e12_rankckpt `
  --model lstm `
  --epochs 12 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --patience 5 `
  --checkpoint-metric val_mean_pairwise_order_accuracy `
  --group-column beatmapset_id `
  --max-notes 3000 `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --huber-delta 0.5 `
  --device cpu `
  --amp off `
  --loader-workers 0 `
  --lstm-embed-dim 8 `
  --lstm-hidden-dim 16 `
  --lstm-layers 1 `
  --lstm-dropout 0.0 `
  --lstm-head-dropout 0.2 `
  --seed 42
```

Validation curve:

| Metric | Start | Final |
| --- | ---: | ---: |
| Validation mean MAE | 0.02094 | 0.01486 |
| Validation pairwise order | 50.00% | 64.14% |

Runtime:

| Metric | Value |
| --- | ---: |
| Average epoch seconds | 32.58 |
| Total epoch seconds | 391.01 |
| Epochs completed | 12 / 12 |
| Best checkpoint epoch | 8 |
| Best checkpoint score | 64.14% validation pairwise |

Holdout test metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.015203 | -0.0062 | -0.4670 | 33.33% | 31.14% |
| acc_std | 0.016971 | 0.0004 | 0.4176 | 65.38% | 41.03% |
| skill_gap | 0.027794 | -0.0179 | -0.3132 | 38.46% | 27.21% |

Checkpoint comparison:

| Run | Checkpoint metric | Best epoch | Mean holdout MAE | Mean holdout pairwise | Mean Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM m3000 tiny e12 | `val_mean_mae` | 12 | 0.019792 | 44.87% | 33.69% |
| LSTM m3000 tiny e12 rankckpt | `val_mean_pairwise_order_accuracy` | 8 | 0.019989 | 45.73% | 33.13% |

Prediction spread check:

| Target | Actual std | Predicted std | Predicted / actual std | Bias |
| --- | ---: | ---: | ---: | ---: |
| mean_acc | 0.024237 | 0.000086 | 0.0035 | -0.000659 |
| acc_std | 0.026729 | 0.000098 | 0.0037 | 0.001886 |
| skill_gap | 0.041247 | 0.000232 | 0.0056 | 0.003700 |

Notes:

- Ranking checkpoint selection only nudged mean holdout pairwise order from
  44.87% to 45.73%, while mean holdout MAE worsened slightly from 0.019792 to
  0.019989.
- The checkpoint metric is not the main blocker. `mean_acc` and `skill_gap`
  still rank poorly, and predicted spread remains near constant.
- Future sequence work should change training signal/model capacity/data
  coverage rather than only selecting checkpoints by validation pairwise order.

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

### Tabular Forest Weighted Seed Stability Check

Command shape:

```powershell
$seeds = 7,13,42,99,123
foreach ($seed in $seeds) {
  .\.venv\Scripts\python.exe -m mania_difficulty.train `
    --labels data\processed\labels_pilot_top100.csv `
    --sequences data\processed\sequences_pilot `
    --run-name "pilot_top100_forest_core_pairwise_weighted_stability_seed${seed}_real" `
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
    --seed $seed
}
```

5-seed weighted grouped out-of-fold stability:

| Seed | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement | Targets beating baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.025605 | -0.0261 | 0.2227 | 57.24% | 7.09% | 3 / 3 |
| 13 | 0.024414 | 0.0585 | 0.3292 | 60.42% | 10.85% | 3 / 3 |
| 42 | 0.023189 | 0.0925 | 0.4330 | 64.46% | 14.53% | 3 / 3 |
| 99 | 0.025242 | -0.0297 | 0.1725 | 55.61% | 8.15% | 3 / 3 |
| 123 | 0.025559 | -0.0620 | 0.2012 | 56.26% | 8.82% | 3 / 3 |
| Mean +/- std | 0.024802 +/- 0.000913 | 0.0066 +/- 0.0586 | 0.2717 +/- 0.0965 | 58.80% +/- 3.28% | 9.89% +/- 2.62% | - |

Comparison against unweighted 5-seed forest:

| Run group | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted 5-seed forest | 0.025329 +/- 0.001089 | 0.0088 +/- 0.0713 | 0.2688 +/- 0.1110 | 58.75% +/- 3.82% | 8.00% +/- 3.26% |
| Score-count weighted 5-seed forest | 0.024802 +/- 0.000913 | 0.0066 +/- 0.0586 | 0.2717 +/- 0.0965 | 58.80% +/- 3.28% | 9.89% +/- 2.62% |

Weighted holdout sanity check:

| Seed | Mean holdout MAE | Mean holdout R2 | Mean holdout Pairwise | Mean Improvement | Targets beating baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.034958 | -0.4499 | 45.10% | -10.36% | 0 / 3 |
| 13 | 0.014865 | -2.9235 | 55.24% | 42.99% | 3 / 3 |
| 42 | 0.029944 | -0.5343 | 48.72% | -1.02% | 1 / 3 |
| 99 | 0.033304 | -1.0829 | 39.29% | -8.14% | 0 / 3 |
| 123 | 0.025117 | -0.5982 | 61.22% | 10.58% | 3 / 3 |

Notes:

- Score-count weighting consistently reduced grouped-CV MAE in all five seeds
  and reduced seed-to-seed MAE variance.
- Ranking did not meaningfully change: weighted and unweighted mean pairwise
  order are both about 58.8%.
- Keep weighting enabled when optimizing reliability/MAE on this pilot dataset,
  but do not expect it to solve ranking without better labels or more maps.

### Tabular Forest Weighted MAE Sweep

Code change:

- `sweep_forest` now passes fold-level `sample_weight` into sklearn `fit(...)`
  when `--sample-weight-column` is set. Before this change, weighted forest
  training used sample weights, but forest parameter sweeps did not.

Command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.sweep_forest `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --out-dir outputs\forest_sweep_pilot_top100_weighted_mae_real `
  --max-notes 7000 `
  --group-column beatmapset_id `
  --cv-folds 5 `
  --seed 42 `
  --trees 100,200,500,800 `
  --min-samples-leaf 1,2,4 `
  --max-features sqrt,0.75,1.0 `
  --feature-sets core `
  --selection-metric mean_mae `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --workers -1
```

Top weighted MAE sweep rows:

| Candidate | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| core_trees100_leaf2_featsqrt | 0.023064 | 0.1016 | 0.4303 | 64.49% | 14.96% |
| core_trees200_leaf2_featsqrt | 0.023189 | 0.0925 | 0.4330 | 64.46% | 14.53% |
| core_trees800_leaf2_featsqrt | 0.023328 | 0.0916 | 0.4189 | 64.16% | 13.99% |
| core_trees500_leaf2_featsqrt | 0.023361 | 0.0917 | 0.4177 | 64.02% | 13.87% |
| core_trees100_leaf2_feat0.75 | 0.023594 | 0.0598 | 0.4159 | 64.07% | 13.01% |

Best-candidate training command:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.train `
  --labels data\processed\labels_pilot_top100.csv `
  --sequences data\processed\sequences_pilot `
  --run-name pilot_top100_forest_core_weighted_mae_best_real `
  --model tabular_forest `
  --feature-set core `
  --forest-trees 100 `
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

Best-candidate 5-fold grouped out-of-fold metrics:

| Target | MAE | R2 | Spearman | Pairwise | MAE improvement vs train-mean baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_acc | 0.017209 | 0.1011 | 0.4598 | 65.43% | 13.45% |
| acc_std | 0.022642 | 0.1059 | 0.3881 | 63.11% | 17.39% |
| skill_gap | 0.029340 | 0.0979 | 0.4428 | 64.94% | 14.05% |

Comparison against the previous weighted 200-tree seed-42 run:

| Run | Mean CV MAE | Mean CV R2 | Mean CV Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: |
| Weighted 200-tree seed 42 | 0.023189 | 0.0925 | 64.46% | 14.53% |
| Weighted 100-tree seed 42 | 0.023064 | 0.1016 | 64.49% | 14.96% |

Notes:

- The weighted MAE sweep found a slightly smaller 100-tree forest that improves
  seed-42 CV MAE and R2 without changing pairwise order meaningfully.
- The holdout split for the best candidate is still noisy and loses to simple
  baselines, so this is only a seed-42 CV tuning candidate.
- The next section checks whether that seed-42 result survives a multi-seed
  stability run.

### Tabular Forest Weighted 100-Tree Seed Stability

Command:

```powershell
$seeds = 7,13,42,99,123
foreach ($seed in $seeds) {
  .\.venv\Scripts\python.exe -m mania_difficulty.train `
    --labels data\processed\labels_pilot_top100.csv `
    --sequences data\processed\sequences_pilot `
    --run-name "pilot_top100_forest_core_weighted100_stability_seed${seed}_real" `
    --model tabular_forest `
    --feature-set core `
    --forest-trees 100 `
    --forest-min-samples-leaf 2 `
    --forest-max-features sqrt `
    --cv-folds 5 `
    --group-column beatmapset_id `
    --max-notes 7000 `
    --sample-weight-column score_count `
    --sample-weight-min 0.25 `
    --sample-weight-max-value 100 `
    --workers -1 `
    --seed $seed
}
```

5-seed grouped out-of-fold stability, averaged across the three targets:

| Seed | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement | Targets beating train-mean baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.025803 | -0.0326 | 0.2190 | 57.00% | 6.39% | 3 / 3 |
| 13 | 0.024175 | 0.0750 | 0.3700 | 61.67% | 11.73% | 3 / 3 |
| 42 | 0.023064 | 0.1016 | 0.4303 | 64.49% | 14.96% | 3 / 3 |
| 99 | 0.025499 | -0.0507 | 0.1609 | 55.19% | 7.20% | 3 / 3 |
| 123 | 0.026067 | -0.0732 | 0.1695 | 55.53% | 7.03% | 3 / 3 |

Weighted 100-tree versus weighted 200-tree stability:

| Run family | Seeds | Mean CV MAE | Mean CV R2 | Mean CV Spearman | Mean CV Pairwise | Mean Improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weighted 200-tree | 5 | 0.024802 +/- 0.000913 | 0.0066 +/- 0.0586 | 0.2717 +/- 0.0965 | 58.80% +/- 3.28% | 9.89% +/- 2.62% |
| Weighted 100-tree | 5 | 0.024921 +/- 0.001134 | 0.0040 +/- 0.0705 | 0.2699 +/- 0.1098 | 58.78% +/- 3.68% | 9.47% +/- 3.34% |

Seed-by-seed delta for weighted 100-tree minus weighted 200-tree:

| Seed | MAE delta | Spearman delta | Pairwise delta | Improvement delta |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000198 | -0.0037 | -0.24% | -0.70% |
| 13 | -0.000240 | 0.0408 | 1.25% | 0.88% |
| 42 | -0.000125 | -0.0027 | 0.03% | 0.44% |
| 99 | 0.000257 | -0.0116 | -0.42% | -0.94% |
| 123 | 0.000508 | -0.0317 | -0.72% | -1.79% |

Notes:

- The 100-tree candidate was better for seed 42, but the 5-seed average does
  not beat the weighted 200-tree stability baseline.
- Weighted 100-tree also has wider seed-to-seed variation on MAE, R2,
  Spearman, pairwise order, and improvement percentage.
- Keep the weighted 200-tree forest as the current small-data tabular baseline;
  treat the 100-tree result as a useful warning against promoting a single-seed
  tuning win.

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
- `outputs\forest_sweep_pilot_top100_weighted_mae_real\sweep_report.html`
- `outputs\neural_sweep_pilot_top100_summary_pairwise_real\neural_sweep_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_clean\run_report.html`
- `outputs\runs\pilot_top100_summary_cpu_real_m7000\run_report.html`
- `outputs\runs\pilot_top100_summary_pairwise_sweep_real_summary_h128_do0p1_lr0p001_wd0p0001_bs8_hd0p5\run_report.html`
- `outputs\runs\pilot_top100_summary_stability_seed{7,13,42,99,123}_real\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m1200\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m3000_tiny_e3\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m3000_tiny_e12\run_report.html`
- `outputs\runs\pilot_top100_lstm_cpu_real_m3000_tiny_e12_rankckpt\run_report.html`
- `outputs\runs\pilot_top100_forest_core_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_best_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_stability_seed{7,13,42,99,123}_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_weighted_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_pairwise_weighted_stability_seed{7,13,42,99,123}_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_weighted_mae_best_real\run_report.html`
- `outputs\runs\pilot_top100_forest_core_weighted100_stability_seed{7,13,42,99,123}_real\run_report.html`
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
read the seed 42 run as optimistic. The unweighted 5-seed grouped-CV estimate is
0.02533 +/- 0.00109 mean MAE and 58.75% +/- 3.82% pairwise order. Score-count
weighting improves this to 0.02480 +/- 0.00091 mean MAE and 58.80% +/- 3.28%
pairwise order, so use weighting for reliability/MAE comparisons while treating
ranking as essentially unchanged. A weighted seed-42 MAE sweep found a
100-tree candidate with slightly better CV MAE than the weighted 200-tree run,
but the 5-seed stability check did not confirm it: weighted 100-tree averaged
0.02492 +/- 0.00113 mean MAE and 58.78% +/- 3.68% pairwise order. Keep the
weighted 200-tree forest as the current comparison baseline.
Treat the CPU LSTM runs as pipeline/performance calibration, not final quality
evidence. The m3000 tiny 12-epoch run improved holdout mean MAE to 0.01979 and
all three targets beat train-mean/difficulty-rating MAE baselines, but holdout
pairwise order was only 44.87% and predictions still collapse to a near-constant
band. Longer sequence training helps calibration/MAE, but it does not yet solve
ranking or spread.
Switching the same tiny LSTM to a validation-pairwise checkpoint only improved
holdout mean pairwise from 44.87% to 45.73% and slightly worsened MAE, so weak
ranking is not mainly a checkpoint-selection issue.

Next training iteration:

1. Prefer `feature-set=core`, 200 trees, leaf 2, and `sqrt` max features for
   tabular pilot comparisons, but report multi-seed CV averages instead of a
   single seed 42 score. Enable `score_count` weighting when the comparison is
   about MAE/reliability rather than pure rank order.
2. Do not adopt the weighted 100-tree candidate on the current 93-map pilot
   dataset; it is slightly worse and less stable than weighted 200-tree across
   five seeds.
3. Treat `summary_hidden_dim=128`, `summary_dropout=0.1`, `lr=0.001`, and
   effective batch size 16 as a local smoke/tuning baseline only; require
   multi-seed or grouped-CV evidence before trusting summary quality.
4. Increase `MAX_NOTES` above 3000 for real pilot/Colab runs when memory allows,
   because 15.05% of pilot maps exceed 3000 notes.
5. Run Colab/GPU LSTM against the same pilot dataset with `MAX_NOTES >= 3000`
   and preferably near 7000 if memory allows. Start from the m3000 tiny e12 CPU
   result as a learning-rate/runtime reference, but on GPU use a larger sequence
   model or ranking-aware tuning because the tiny CPU run still collapses to
   near-constant predictions and weak holdout ranking. Do not rely on
   `val_mean_pairwise_order_accuracy` checkpoint selection alone; it did not
   materially fix holdout ranking.
6. Add more top100 maps before claiming model quality; 93 maps is still a pilot
   dataset.
