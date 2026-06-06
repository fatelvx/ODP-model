from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def archive_name(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    return relative.as_posix()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(file for file in path.rglob("*") if file.is_file())


def package_artifacts(
    paths: Iterable[Path],
    out_zip: Path,
    *,
    manifest_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    root = Path.cwd() if root is None else root
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = out_zip.with_name("artifact_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_zip_resolved = out_zip.resolve()
    manifest_resolved = manifest_path.resolve()

    included_roots: list[dict[str, object]] = []
    missing: list[str] = []
    files: list[Path] = []
    seen_archive_names: set[str] = set()

    for raw_path in paths:
        path = Path(raw_path)
        display_path = archive_name(path, root)
        if not path.exists():
            missing.append(display_path)
            continue

        artifact_files = iter_files(path)
        total_bytes = sum(file.stat().st_size for file in artifact_files)
        included_roots.append(
            {
                "path": display_path,
                "kind": "directory" if path.is_dir() else "file",
                "file_count": len(artifact_files),
                "size_bytes": total_bytes,
            }
        )
        for file in artifact_files:
            if file.resolve() == out_zip_resolved:
                continue
            name = archive_name(file, root)
            if name in seen_archive_names:
                continue
            seen_archive_names.add(name)
            files.append(file)

    manifest_name = archive_name(manifest_path, root)
    if manifest_name not in seen_archive_names:
        seen_archive_names.add(manifest_name)
        files.append(manifest_path)

    payload_files = [file for file in files if file.resolve() != manifest_resolved]
    payload_size_bytes = sum(file.stat().st_size for file in payload_files)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_zip": archive_name(out_zip, root),
        "root": str(root),
        "included": included_roots,
        "missing": missing,
        "file_count": len(files),
        "payload_file_count": len(payload_files),
        "total_size_bytes": payload_size_bytes,
        "manifest_path": manifest_name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, archive_name(file, root))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package training artifacts into a zip with a manifest.")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--out-zip", type=Path, default=Path("colab_top100_outputs.zip"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    manifest = package_artifacts(
        args.paths,
        args.out_zip,
        manifest_path=args.manifest,
        root=args.root,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out_zip}")


if __name__ == "__main__":
    main()
