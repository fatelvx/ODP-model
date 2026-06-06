# Player-Feel V1 Implementation Notes

Date: 2026-06-06

## Decision

The project is shifting from pure top100 regression toward a 4K psychometric
player-feel model. Top100 scores remain useful as weak proxy data, but the V1
training target is human pairwise feel: which whole chart or segment feels
harder for a named player stage.

The annotation design explicitly handles rater coverage limits. For comparisons
below or above the rater's reliable range, use `uncertain`, `out_of_range`,
`skip`, or `tie` in `harder_choice`; those rows are retained but ignored by the
ranker. `confidence` is used as sample weight for judged rows.

## Implemented

- Added fixed 4K player stages in `data/player_stages_4k.csv`.
- Added an empty fillable annotation template in
  `data/annotations/player_feel_pairs.csv`.
- Added source-integrity guardrails for key mode, osu!mania mode, HP/OD/AR, and
  optional raw `.osu` metadata cross-checks.
- Added player-feel curve extraction with pressure columns for reading, speed,
  stamina, jack, chord, LN, and accuracy.
- Added pair candidate generation for whole-map and segment comparisons.
- Added a confidence-weighted pairwise logistic ranker and HTML/CSV/JSON report
  outputs.
- Added a local browser labeler for pairwise judgments. It loads the generated
  pair CSV, filters by stage/scope/status, supports `A`, `B`, `tie`,
  `uncertain`, `out_of_range`, and `skip`, and writes
  `confidence`/`reason_tags`/`notes` back to the same file.

## Smoke Training Evidence

Command family:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.make_synthetic_dataset --maps 64 --out data\processed\synthetic_player_feel_v1 --seed 607
.\.venv\Scripts\python.exe -m mania_difficulty.tools.player_feel_curve --labels data\processed\synthetic_player_feel_v1\labels.csv --sequences data\processed\synthetic_player_feel_v1\sequences --out-dir outputs\player_feel_v1_smoke\curves
.\.venv\Scripts\python.exe -m mania_difficulty.tools.generate_player_feel_pairs --labels data\processed\synthetic_player_feel_v1\labels.csv --sequences data\processed\synthetic_player_feel_v1\sequences --player-stages data\player_stages_4k.csv --out outputs\player_feel_v1_smoke\fake_pairs.csv --max-pairs 100 --stage-ids beginner,intermediate,dan_ready
.\.venv\Scripts\python.exe -m mania_difficulty.tools.train_feel_ranker --judgments outputs\player_feel_v1_smoke\fake_judgments.csv --summary outputs\player_feel_v1_smoke\curves\player_feel_summary.csv --curves outputs\player_feel_v1_smoke\curves\player_feel_curve.csv --player-stages data\player_stages_4k.csv --out-dir outputs\player_feel_v1_smoke\ranker --test-size 0.25 --seed 607
```

Smoke result:

| Metric | Value |
| --- | ---: |
| Usable fake judgments | 100 |
| Feature count | 36 |
| Train rows | 75 |
| Holdout rows | 25 |
| Train agreement | 94.67% |
| Holdout agreement | 76.00% |

This proves the V1 ranking pipeline trains and evaluates end to end. It does
not prove true human-feel quality because the smoke labels are synthetic.

## Annotation UI

Start the local labeler with:

```powershell
.\.venv\Scripts\python.exe -m mania_difficulty.tools.serve_player_feel_labeler `
  --pairs outputs\player_feel_v1_pilot_real\player_feel_pairs_to_label.csv `
  --host 127.0.0.1 `
  --port 8765
```

Then open `http://127.0.0.1:8765`. The UI is intentionally optimized for the
current rater-coverage problem: if a pair is too easy to feel accurately or too
hard to judge from experience, mark `out_of_range` or `uncertain` instead of
forcing a fake answer.

HTTP smoke check passed against a temporary copy of the real pilot pair CSV:

| Check | Result |
| --- | --- |
| `/health` | ok |
| `/api/state?status=open` | returned the first open pair |
| `/api/save` | saved `out_of_range` to the temporary CSV |

## Real Pilot Artifacts

Generated from the 93-map 4K pilot dataset:

- `outputs\player_feel_v1_pilot_real\curves\player_feel_curve.csv`
- `outputs\player_feel_v1_pilot_real\curves\player_feel_summary.csv`
- `outputs\player_feel_v1_pilot_real\curves\player_feel_curves.png`
- `outputs\player_feel_v1_pilot_real\player_feel_pairs_to_label.csv`

Pilot dominant-skill distribution after excluding aggregate accuracy/stamina
from the primary pattern label:

| Dominant skill | Count |
| --- | ---: |
| chord | 44 |
| speed | 30 |
| LN | 10 |
| jack | 9 |

## Next Acceptance Gate

Fill at least 50 real human judgments in
`outputs\player_feel_v1_pilot_real\player_feel_pairs_to_label.csv`, then train
the real ranker. V1 starts to count as "has player feel" only if held-out
pairwise agreement reaches at least 65%; below that, the pipeline is working but
the model is not yet trusted.
