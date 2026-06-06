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

- `history.csv`: train/val loss per epoch
- `learning_curve.png`: growth curve
- `predictions.csv`: actual model outputs on the test split
- `human_review.csv`: maps worth checking by hand
- `human_pair_review.csv`: map pairs where model and top100 proxy rank difficulty differently
- `human_pair_judgment_template.csv`: fillable CSV for scoring model/proxy agreement with human judgment
- `error_slices.csv`: mean-accuracy error grouped by metadata bins such as score count and note count
- `prediction_scatter.png`: predicted vs actual plots
- `cv_metrics.json`: K-fold out-of-fold metrics when `--cv-folds` is enabled
- `cv_human_pair_review.csv`: cross-validation pairwise disagreements when `--cv-folds` is enabled
- `cv_human_pair_judgment_template.csv`: fillable out-of-fold human judgment CSV
- `cv_error_slices.csv`: cross-validation error slices when `--cv-folds` is enabled
- `cv_prediction_scatter.png`: cross-validation predicted vs actual plots
- `metrics.json`: MAE and R2 per target
- `best_model.pt`: checkpoint for prediction

Metrics include a train-mean baseline when available:

- `baseline_mae`: MAE from predicting the training mean for that target
- `mae_improvement_vs_baseline`: positive means the model beat that baseline
- `mae_improvement_pct`: the same improvement as a percent
- `spearman`: rank correlation between predicted and observed proxy values
- `pairwise_order_accuracy`: percent of map pairs ranked in the same order by model and proxy

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
spending GPU time.

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
  --model lstm `
  --run-name lstm_top100_baseline
```

For local CPU pilots, use `--model summary`. It is much faster and is meant to
prove the data/label signal before spending GPU time. Use `--model
tabular_forest` as the small-data baseline and `--model lstm` in Colab or on a
GPU machine.

Neural model architecture knobs are CLI-controlled and saved in each run report:

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

Tune summary/LSTM neural parameters with a holdout sweep:

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
  --lrs 0.001,0.0005 `
  --weight-decays 0.0001 `
  --lstm-embed-dims 32,64 `
  --lstm-hidden-dims 64,128 `
  --lstm-layers 1,2 `
  --lstm-dropouts 0.1,0.2 `
  --lstm-head-dropouts 0.2 `
  --selection-metric mean_pairwise_order_accuracy `
  --max-candidates 4 `
  --loader-workers 2 `
  --device cuda
```

The neural sweep writes `neural_sweep_summary.csv`,
`neural_sweep_details.csv`, `best_params.json`, and
`neural_sweep_report.html`. Use `--selection-metric mean_mae` for absolute
prediction error, or `--selection-metric mean_pairwise_order_accuracy` when the
priority is choosing the same harder/easier direction a human would compare.
On Colab or another Linux GPU runtime, use `--loader-workers 2` to keep the GPU
fed. On local Windows CPU pilots, leave it at the default `0`.

Compare runs:

```powershell
python -m mania_difficulty.tools.compare_runs `
  outputs/runs/forest_top100_baseline `
  outputs/runs/lstm_top100_baseline
```

Build one dashboard that links the audit, sweeps, comparison, metrics, plots,
human-review files, and any filled human judgment score tables:

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

After filling `human_pair_judgment_template.csv` or
`cv_human_pair_judgment_template.csv`, score whether the model or top100 proxy
matched your human judgment. Rebuild the dashboard after filling the CSV to see
the same score table inside the run card:

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

VS Code should recommend the official `google.colab` extension when this repo is
opened.

## Notes

- The fetch scripts cache files and skip data that already exists.
- Score fetching sleeps between requests by default.
- `fetch_maps.py` uses osu!'s current `beatmapsets/search` API shape and filters
  for 4K mania beatmaps from the returned beatmapsets.
- The model starts with LSTM because it is easier to train and debug. Transformer
  attention analysis can come after the baseline is trustworthy.
