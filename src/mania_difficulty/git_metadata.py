from __future__ import annotations

import subprocess
from pathlib import Path


def git_command_output(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def git_environment_metadata(cwd: Path | None = None) -> dict[str, object]:
    repo_cwd = Path.cwd() if cwd is None else Path(cwd)
    commit = git_command_output(["rev-parse", "--short", "HEAD"], repo_cwd)
    if not commit:
        return {
            "git_commit": "",
            "git_branch": "",
            "git_dirty": "",
            "git_status_entries": "",
        }

    branch = git_command_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_cwd)
    if branch == "HEAD":
        branch = ""
    status = git_command_output(["status", "--short"], repo_cwd)
    status_entries = len(status.splitlines()) if status else 0
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": status_entries > 0,
        "git_status_entries": status_entries,
    }
