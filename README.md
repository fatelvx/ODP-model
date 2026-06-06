# osu!mania Difficulty Prediction Model

Raw-note osu!mania 4K difficulty prediction from leaderboard accuracy data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatelvx/ODP-model/blob/main/notebooks/colab_train.ipynb)

This repo is set up as a learning-friendly baseline: every training run saves
human-readable outputs so we can see whether the model is improving instead of
only staring at terminal logs.

## What This Predicts

The first target is a leaderboard proxy, not the full playerbase:

- `mean_acc`: average accuracy among fetched scores
- `acc_std`: accuracy spread
- `skill_gap`: average top 10% accuracy minus average bottom 50% accuracy

The osu! API only exposes the visible score data we can reasonably fetch, so the
labels should be read as "top leaderboard performance descriptors".

## Quick Smoke Test Without osu! API

Local setup uses Python 3.12 in this repo because the default Python on this
machine is 3.14 and did not have torch installed.

Install dependencies with uv:

```powershell
uv venv --python 3.12 .venv
uv pip install -e .
```

Or with regular Python/pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a tiny synthetic dataset:

```powershell
python -m mania_difficulty.tools.make_synthetic_dataset --maps 96 --out data/processed/synthetic
```

Train a fast CPU baseline for a short run:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/synthetic/labels.csv `
  --sequences data/processed/synthetic/sequences `
  --epochs 8 `
  --batch-size 16 `
  --model summary `
  --run-name synthetic_smoke
```

For small real datasets, also train the tabular forest baseline. It is usually
more stable than a neural sequence model before we have many labels:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/synthetic/labels.csv `
  --sequences data/processed/synthetic/sequences `
  --model tabular_forest `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --run-name synthetic_forest_smoke
```

Tune forest parameters with group-aware CV:

```powershell
python -m mania_difficulty.tools.sweep_forest `
  --labels data/processed/synthetic/labels.csv `
  --sequences data/processed/synthetic/sequences `
  --out-dir outputs/forest_sweep_synthetic `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --trees 200,500 `
  --min-samples-leaf 1,2,4 `
  --max-features sqrt,0.75 `
  --feature-sets core,burst `
  --selection-metric mean_pairwise_order_accuracy
```

The sweep writes `sweep_summary.csv`, `sweep_details.csv`,
`best_params.json`, and `sweep_report.html`.

Feature sets:

- `core`: stable summary baseline features
- `burst`: adds peak density, burst ratio, jack, trill-like, and chord-burst features

Open the generated report:

```powershell
start outputs/runs/synthetic_smoke/run_report.html
```

Each run saves:

- `history.csv`: train/val loss, validation MAE, validation pairwise order, epoch seconds, LR, and CUDA peak memory per epoch
- `learning_curve.png`: loss curve plus validation MAE / pairwise-order curve when available
- `run_report.html`: includes Model Verdict, Training Health, Training Performance, and Worst Error Slices tables for quick quality, validation metrics, loss, speed, LR, memory, runtime device, baseline wins, and fragile metadata-bin checks
- `predictions.csv`: actual model outputs on the test split
- `prediction_summary.csv`: per-target actual mean, predicted mean, bias, MAE, max error, and spread
- `prediction_rankings.csv`: predicted hardest, predicted easiest, and largest-error maps
- `human_review.csv`: maps worth checking by hand
- `human_pair_review.csv`: map pairs where model and top100 proxy rank difficulty differently
- `human_pair_judgment_template.csv`: fillable CSV for scoring model/proxy agreement with human judgment
- `error_slices.csv`: mean-accuracy error grouped by metadata bins such as score count and note count
- `prediction_scatter.png`: predicted vs actual plots
- `embedding_projection.csv`, `embedding_projection.png`, `embedding_report.html`: 2D model embedding projection when `project_embeddings` is run
- `attention_map.csv`, `attention_map.png`, `attention_report.html`: Transformer note-level attention for one selected map when `attention_map` is run
- `cv_metrics.json`: K-fold out-of-fold metrics when `--cv-folds` is enabled
- `cv_human_pair_review.csv`: cross-validation pairwise disagreements when `--cv-folds` is enabled
- `cv_human_pair_judgment_template.csv`: fillable out-of-fold human judgment CSV
- `cv_error_slices.csv`: cross-validation error slices when `--cv-folds` is enabled
- `cv_prediction_scatter.png`: cross-validation predicted vs actual plots
- `cv_prediction_summary.csv`: out-of-fold per-target bias and error scale
- `cv_prediction_rankings.csv`: cross-validation predicted hardest, easiest, and largest-error maps
- `metrics.json`: MAE and R2 per target
- `best_model.pt`: checkpoint for prediction, selected by `--checkpoint-metric` for neural runs
- `last_checkpoint.pt`: neural training state for `--resume`

Model Verdict includes `Next Action`, a conservative recommendation based on
baseline wins and ranking signal. It is meant to answer whether the next step
should be label/human review, feature/model tuning, or keeping the run as the
current baseline.

Metrics include a train-mean baseline when available:

- `baseline_mae`: MAE from predicting the training mean for that target
- `mae_improvement_vs_baseline`: positive means the model beat that baseline
- `mae_improvement_pct`: the same improvement as a percent
- `spearman`: rank correlation between predicted and observed proxy values
- `pairwise_order_accuracy`: percent of map pairs ranked in the same order by model and proxy

When labels include osu!'s `difficulty_rating`, training also fits a simple
linear `difficulty_rating -> target` baseline on the training split and reports
`difficulty_rating_baseline_mae` plus improvement vs that baseline. This tells
us whether the sequence model is beating a basic existing difficulty signal,
not just the train-mean baseline. Run reports and dashboards summarize this as
`Targets Beating Difficulty Rating` and `Mean Difficulty Rating Improvement`
when the baseline is available.

When the label file has `beatmapset_id`, train/validation/test and
cross-validation splits keep the same beatmapset in only one split. This avoids
overstating performance by testing on another difficulty from a mapset the
model already saw.

## Real Data Pipeline

Set osu! API credentials first:

```powershell
$env:OSU_CLIENT_ID="your_client_id"
$env:OSU_CLIENT_SECRET="your_client_secret"
```

Fetch ranked 4K mania map metadata:

```powershell
python -m mania_difficulty.data.fetch_maps --target 2000 --out data/raw/beatmaps.csv
```

Download `.osu` files through the configured mirror:

```powershell
python -m mania_difficulty.data.fetch_osu_files `
  --maps data/raw/beatmaps.csv `
  --out-dir data/raw/osu
```

Parse note tensors:

```powershell
python -m mania_difficulty.data.parse_notes `
  --maps data/raw/beatmaps.csv `
  --osu-dir data/raw/osu `
  --out-dir data/processed/sequences
```

Fetch score labels:

```powershell
python -m mania_difficulty.data.fetch_scores `
  --maps data/raw/beatmaps.csv `
  --out data/processed/labels.csv `
  --min-scores 30
```

Audit label/sequence coverage before training:

```powershell
python -m mania_difficulty.tools.audit_dataset `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --out-dir outputs/dataset_audit_top100
```

The audit writes `dataset_audit.json`, `missing_sequences.csv`,
`dataset_distributions.png`, and `dataset_audit.html` so we can catch missing
parsed files, weak label coverage, or narrow target distributions before
spending GPU time. The audit also reports Label Reliability from `score_count`,
including the full top100 rate and maps below the low-score threshold.

Train a small-data forest baseline:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --model tabular_forest `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --run-name forest_top100_baseline
```

Run the same forest sweep on real labels before choosing final forest
parameters:

```powershell
python -m mania_difficulty.tools.sweep_forest `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --out-dir outputs/forest_sweep_top100 `
  --cv-folds 5 `
  --group-column beatmapset_id `
  --trees 200,500 `
  --min-samples-leaf 1,2,4 `
  --max-features sqrt,0.75 `
  --feature-sets core,burst `
  --selection-metric mean_pairwise_order_accuracy
```

Train the sequence model:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --epochs 50 `
  --batch-size 32 `
  --grad-clip-norm 1.0 `
  --checkpoint-metric val_mean_mae `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --model lstm `
  --run-name lstm_top100_baseline
```

For neural runs, `--checkpoint-metric` controls which epoch becomes
`best_model.pt` and drives early stopping. Use `val_mean_mae` for the most
readable validation error, or `val_mean_pairwise_order_accuracy` when the
priority is ranking map pairs in the same harder/easier direction.
Neural runs clip gradient norm at `--grad-clip-norm 1.0` by default to reduce
unstable updates on small/noisy top100 labels; set it to `0` to disable clipping
or lower it if training spikes.

For neural runs on top100 labels, `--sample-weight-column score_count`
downweights maps with fewer visible scores. With the defaults above, 100 scores
maps to weight 1.0, 50 scores maps to 0.5, and very low-count maps bottom out at
0.25 instead of steering the model as strongly as full top100 labels. Run
reports and dashboard decision summaries include the train split's mean sample
weight and downweighted rate so this effect is visible after training.

For local CPU pilots, use `--model summary`. It is much faster and is meant to
prove the data/label signal before spending GPU time. Use `--model
tabular_forest` as the small-data baseline and `--model lstm` in Colab or on a
GPU machine. After the LSTM route is stable, `--model transformer` is available
as the heavier sequence baseline for later attention analysis.

Neural model architecture knobs are CLI-controlled and saved in each run report:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --model transformer `
  --run-name transformer_top100_pilot `
  --max-notes 3000 `
  --transformer-embed-dim 64 `
  --transformer-heads 4 `
  --transformer-layers 3 `
  --transformer-ff-dim 256 `
  --transformer-dropout 0.1 `
  --transformer-head-dropout 0.2
```

For LSTM tuning:

```powershell
python -m mania_difficulty.train `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --model lstm `
  --run-name lstm_top100_tuned `
  --lstm-embed-dim 64 `
  --lstm-hidden-dim 128 `
  --lstm-layers 2 `
  --lstm-dropout 0.2 `
  --lstm-head-dropout 0.3
```

Tune summary/LSTM/Transformer neural parameters with a holdout sweep. Keep
`--models lstm` for the normal Colab path; include `transformer` only when you
want to spend extra GPU time:

```powershell
python -m mania_difficulty.tools.sweep_neural `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --out-dir outputs/neural_sweep_top100 `
  --run-prefix neural_sweep_top100 `
  --models lstm `
  --epochs 30 `
  --patience 6 `
  --batch-size 32 `
  --grad-accum-steps 1 `
  --grad-clip-norm 1.0 `
  --checkpoint-metric val_mean_mae `
  --sample-weight-column score_count `
  --sample-weight-min 0.25 `
  --sample-weight-max-value 100 `
  --lrs 0.001,0.0005 `
  --weight-decays 0.0001 `
  --lstm-embed-dims 32,64 `
  --lstm-hidden-dims 64,128 `
  --lstm-layers 1,2 `
  --lstm-dropouts 0.1,0.2 `
  --lstm-head-dropouts 0.2 `
  --transformer-embed-dims 32,64 `
  --transformer-heads 4 `
  --transformer-layers 1,2 `
  --transformer-ff-dims 128,256 `
  --selection-metric mean_pairwise_order_accuracy `
  --max-candidates 4 `
  --loader-workers 2 `
  --amp auto `
  --device cuda
```

The neural sweep writes `neural_sweep_summary.csv`,
`neural_sweep_details.csv`, `best_params.json`, and
`neural_sweep_report.html`. Use `--selection-metric mean_mae` for absolute
prediction error, or `--selection-metric mean_pairwise_order_accuracy` when the
priority is choosing the same harder/easier direction a human would compare.
On Colab or another Linux GPU runtime, use `--loader-workers 2` to keep the GPU
fed. `--amp auto` enables mixed precision on CUDA and stays off on CPU. If a GPU
run is out of memory, lower `--batch-size` and raise `--grad-accum-steps 2` or
`4`; the effective batch size is `batch-size * grad-accum-steps`. On local
Windows CPU pilots, leave loader workers at the default `0`.

Compare runs:

```powershell
python -m mania_difficulty.tools.compare_runs `
  outputs/runs/forest_top100_baseline `
  outputs/runs/lstm_top100_baseline
```

Project the model's internal representation to 2D. For neural checkpoints this
uses the pre-head embedding; for `tabular_forest` it uses the summary feature
vector. The resulting plot helps check whether maps naturally cluster by
difficulty shape:

```powershell
python -m mania_difficulty.tools.project_embeddings `
  --checkpoint outputs/runs/lstm_top100_baseline/best_model.pt `
  --labels data/processed/labels.csv `
  --sequences data/processed/sequences `
  --method pca `
  --color-target mean_acc
```

For a Transformer checkpoint, visualize note-level attention on one selected
map. The output is a column/time plot plus a CSV of per-note attention weights:

```powershell
python -m mania_difficulty.tools.attention_map `
  --checkpoint outputs/runs/transformer_top100_pilot/best_model.pt `
  --beatmap-id 123456 `
  --sequences data/processed/sequences
```

Build one dashboard that links the audit, sweeps, comparison, metrics, plots,
human-review files, worst error slices, and any filled human judgment score
tables:

```powershell
python -m mania_difficulty.tools.build_dashboard `
  outputs/runs/forest_top100_baseline `
  outputs/runs/lstm_top100_baseline `
  --audit-dir outputs/dataset_audit_top100 `
  --forest-sweep-dir outputs/forest_sweep_top100 `
  --neural-sweep-dir outputs/neural_sweep_top100 `
  --comparison-html outputs/run_comparison.html `
  --out-html outputs/dashboard.html
```

Start with the dashboard's `Run Decision Summary` table when comparing runs.
It condenses each run into baseline wins, ranking quality, weakest target, and
the generated `Next Action`; it also keeps device, AMP, effective batch size,
gradient clip norm, epochs completed, and stop reason visible so Colab GPU runs
are easier to compare. It also includes learning-curve health such as best/final
validation MAE, best/final pairwise order, overfit signal, average epoch seconds,
and peak CUDA memory. It also reads `prediction_summary.csv` and
`cv_prediction_summary.csv` into calibration fields such as mean absolute bias,
worst biased target, and predicted-vs-actual spread ratio; low spread ratio is a
warning that the model may be collapsing toward average predictions. The
`training_adjustment` column turns those signals into a conservative next-run
tuning hint, such as adding regularization, running a small sweep, moving neural
training to CUDA, raising model capacity when predictions are too compressed, or
lowering batch size when memory is tight. If a judgment template has been
filled, the summary also shows human judgment coverage plus model/proxy
agreement rates so runs can be compared against manual harder/easier calls.
Then use the run cards below it for plots and human-review files. The dashboard
command also writes
`run_decision_summary.csv` beside the HTML so the same summary can be sorted,
saved, or included in Colab output downloads.

After filling `human_pair_judgment_template.csv` or
`cv_human_pair_judgment_template.csv`, score whether the model or top100 proxy
matched your human judgment. Rebuild the dashboard after filling the CSV to see
the same score table inside the run card. `human_confidence` is optional; when
filled, the score also reports confidence-weighted agreement rates:

```powershell
python -m mania_difficulty.tools.human_judgments score `
  --judgments outputs/runs/lstm_top100_baseline/human_pair_judgment_template.csv `
  --out-json outputs/human_judgment_score.json `
  --out-html outputs/human_judgment_score.html
```

Predict one `.osu` file after training:

```powershell
python -m mania_difficulty.predict `
  --checkpoint outputs/runs/lstm_top100_baseline/best_model.pt `
  --osu data/raw/osu/123456.osu
```

## Colab / VS Code Colab Path

If local training is too slow, use `notebooks/colab_train.ipynb`.

Recommended workflow:

1. Open the notebook through the badge above, Google Colab, or the official
   Colab VS Code extension.
2. Enable a GPU runtime.
3. Run the cells to install the project, train, and display the learning curve.

The notebook includes a synthetic smoke path and a real-data path. For real
data, keep API credentials in the notebook session only; do not commit them.
The Check GPU cell sets `TRAIN_DEVICE`, `LOADER_WORKERS`, and `AMP_MODE` once,
then the smoke test, neural sweep, final training, embedding projection, and
attention map cells reuse those settings. If the runtime is still CPU, the cell
prints a warning before real neural training time is wasted.
The real-data path keeps LSTM as the normal Colab default, but if the neural
sweep is changed to include `transformer`, the final run and dashboard use
`colab_{model}_top100` automatically. If Colab reports CUDA out-of-memory,
change `GRAD_ACCUM_STEPS` in the notebook from `1` to `2` or `4`. If a long
final neural run is interrupted, re-run the final training cell with
`RESUME_FINAL_TRAINING = True` to continue from `last_checkpoint.pt`. To survive
a full Colab runtime reset, set `USE_DRIVE_CHECKPOINT_BACKUP = True`; the final
training command will use `--checkpoint-backup-dir` to sync checkpoints to
Google Drive and restore them before resume. The notebook sets
`CHECKPOINT_METRIC = "val_mean_mae"` by default; change it to
`"val_mean_pairwise_order_accuracy"` if you want the best checkpoint chosen by
harder/easier ranking agreement instead of absolute validation error.
When the real-data cells finish, Colab packages the audit, sweeps, run reports,
plots, dashboard, decision summary, and `outputs/colab_artifact_manifest.json`
into `colab_top100_outputs.zip`. The manifest lists included paths and missing
artifacts so downloaded results are easier to verify later. The notebook also
displays the decision summary's calibration columns and each run's
`prediction_summary.csv` or `cv_prediction_summary.csv`, so bias and
prediction-spread warnings are visible before downloading the zip.

VS Code should recommend the official `google.colab` extension when this repo is
opened.

## Notes

- The fetch scripts cache files and skip data that already exists.
- Score fetching sleeps between requests by default.
- `fetch_maps.py` uses osu!'s current `beatmapsets/search` API shape and filters
  for 4K mania beatmaps from the returned beatmapsets.
- The default sequence path stays on LSTM because it is easier to train and
  debug. Transformer training is available for GPU pilots and future attention
  analysis once the baseline is trustworthy.
