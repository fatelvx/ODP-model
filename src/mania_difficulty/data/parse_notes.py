from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


MANIA_PLAYFIELD_WIDTH = 512
HOLD_NOTE_TYPE_BIT = 128
OSU_METADATA_FIELDS = {
    "mode": ("General", "Mode"),
    "circle_size": ("Difficulty", "CircleSize"),
    "hp_drain_rate": ("Difficulty", "HPDrainRate"),
    "overall_difficulty": ("Difficulty", "OverallDifficulty"),
    "approach_rate": ("Difficulty", "ApproachRate"),
}


@dataclass(frozen=True)
class NoteEvent:
    time_ms: int
    column: int
    is_ln: int
    ln_length_ms: int
    chord_size: int = 1


def hitobject_lines(osu_text: str) -> list[str]:
    in_hitobjects = False
    lines: list[str] = []
    for raw_line in osu_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_hitobjects = line == "[HitObjects]"
            continue
        if in_hitobjects:
            lines.append(line)
    return lines


def parse_osu_metadata(osu_text: str) -> dict[str, float]:
    section = ""
    raw_values: dict[tuple[str, str], str] = {}
    for raw_line in osu_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw_values[(section, key.strip())] = value.strip()

    metadata: dict[str, float] = {}
    for output_name, source in OSU_METADATA_FIELDS.items():
        raw_value = raw_values.get(source)
        if raw_value in (None, ""):
            continue
        try:
            metadata[output_name] = float(raw_value)
        except ValueError:
            continue
    if "circle_size" in metadata:
        metadata["keys"] = metadata["circle_size"]
    return metadata


def x_to_column(x: int, keys: int = 4) -> int:
    column = int(x * keys / MANIA_PLAYFIELD_WIDTH)
    return max(0, min(keys - 1, column))


def parse_hitobject(line: str, keys: int = 4) -> NoteEvent | None:
    parts = line.split(",")
    if len(parts) < 5:
        return None

    try:
        x = int(float(parts[0]))
        time_ms = int(float(parts[2]))
        note_type = int(parts[3])
    except ValueError:
        return None

    is_ln = 1 if note_type & HOLD_NOTE_TYPE_BIT else 0
    ln_length_ms = 0
    if is_ln and len(parts) >= 6:
        end_time_raw = parts[5].split(":")[0]
        try:
            ln_length_ms = max(0, int(float(end_time_raw)) - time_ms)
        except ValueError:
            ln_length_ms = 0

    return NoteEvent(
        time_ms=time_ms,
        column=x_to_column(x, keys),
        is_ln=is_ln,
        ln_length_ms=ln_length_ms,
    )


def parse_osu_text(osu_text: str, keys: int = 4) -> list[NoteEvent]:
    events = [
        event
        for line in hitobject_lines(osu_text)
        if (event := parse_hitobject(line, keys)) is not None
    ]
    events.sort(key=lambda event: (event.time_ms, event.column))

    chord_sizes: dict[int, int] = {}
    for event in events:
        chord_sizes[event.time_ms] = chord_sizes.get(event.time_ms, 0) + 1

    return [
        NoteEvent(
            time_ms=event.time_ms,
            column=event.column,
            is_ln=event.is_ln,
            ln_length_ms=event.ln_length_ms,
            chord_size=chord_sizes[event.time_ms],
        )
        for event in events
    ]


def events_to_features(events: list[NoteEvent], keys: int = 4) -> list[list[float]]:
    if not events:
        return []

    max_time = max(events[-1].time_ms, 1)
    previous_time = events[0].time_ms
    features: list[list[float]] = []
    for index, event in enumerate(events):
        delta_ms = 0 if index == 0 else event.time_ms - previous_time
        previous_time = event.time_ms
        features.append(
            [
                event.time_ms / max_time,
                delta_ms / 1000.0,
                event.column / max(1, keys - 1),
                float(event.is_ln),
                event.ln_length_ms / 1000.0,
                event.chord_size / keys,
            ]
        )
    return features


def parse_osu_file(path: Path, keys: int = 4) -> list[list[float]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return events_to_features(parse_osu_text(text, keys), keys)


def read_osu_metadata(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return parse_osu_metadata(text)


def save_sequence(features: list[list[float]], out_path: Path) -> None:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(features, dtype=np.float32)
    np.save(out_path, array)


def read_map_ids(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [row["beatmap_id"] for row in csv.DictReader(file)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse .osu hit objects into note tensors.")
    parser.add_argument("--maps", type=Path, default=Path("data/raw/beatmaps.csv"))
    parser.add_argument("--osu-dir", type=Path, default=Path("data/raw/osu"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sequences"))
    parser.add_argument("--keys", type=int, default=4)
    parser.add_argument("--min-notes", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parsed = 0
    skipped = 0
    for beatmap_id in tqdm(read_map_ids(args.maps), desc="parse"):
        osu_path = args.osu_dir / f"{beatmap_id}.osu"
        out_path = args.out_dir / f"{beatmap_id}.npy"
        if not osu_path.exists():
            skipped += 1
            continue
        features = parse_osu_file(osu_path, args.keys)
        if len(features) < args.min_notes:
            skipped += 1
            continue
        save_sequence(features, out_path)
        parsed += 1

    print(f"Parsed {parsed} maps into {args.out_dir}; skipped {skipped}.")


if __name__ == "__main__":
    main()
