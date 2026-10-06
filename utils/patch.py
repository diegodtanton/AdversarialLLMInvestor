#!/usr/bin/env python3
"""Apply and commit a Git patch on branch 'patch', then return to 'main'.

Save this file as utils/patch.py in your repository. Run from the repository
root or any subdirectory of it. Patch paths are relative to the directory
where you run the command; absolute paths and ~ are also supported.

Examples (from the repository root):
    python utils/patch.py ../alchemist-audit/Alchemist-audit.patch
    python utils/patch.py ../alchemist-audit/Alchemist-audit.patch --check

Requires Python 3.10+ and Git 2.23+ (git switch). No Python packages required.
Requires a local main branch, a Git author identity, and no staged or modified
tracked files. An existing patch branch is reused only if it is already
merged into main; it is fast-forwarded to main before applying the new patch.
Untracked files, including this helper and the patch itself, may remain.
An index-aware preflight rejects collisions with untracked patch targets.

Only the patch changes are staged and committed. The main branch is not
merged into or modified by the patch. Nothing is pushed or stashed.
Normal Git commit hooks and signing settings are respected. If application,
commit, or checkout fails, the script preserves the state for inspection.
--check requires that you are already on main and does not change anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


class PatchError(Exception):
    """An actionable error that should not print a Python traceback."""


def git(
    repo: Path,
    *args: str,
    patch: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise PatchError(detail or f"git {args[0]} failed (exit {result.returncode}).")
    return result


def text(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def repository_root() -> Path:
    if not shutil.which("git"):
        raise PatchError("Git was not found. Install Git and make sure it is on PATH.")
    result = git(Path.cwd(), "rev-parse", "--show-toplevel", check=False)
    if result.returncode:
        raise PatchError("Run this command inside your original Git repository.")
    return Path(text(result.stdout))


def ensure_ready(repo: Path) -> None:
    if git(repo, "rev-parse", "--verify", "HEAD", check=False).returncode:
        raise PatchError("This repository needs an initial commit before creating a patch branch.")

    # Resolve Git's administrative paths correctly for linked worktrees too.
    for marker in (
        "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge",
        "rebase-apply", "sequencer", "BISECT_LOG",
    ):
        administrative_path = Path(text(git(repo, "rev-parse", "--git-path", marker).stdout))
        if not administrative_path.is_absolute():
            administrative_path = repo / administrative_path
        if administrative_path.exists():
            raise PatchError(
                f"An unfinished Git operation is present ({marker}). "
                "Finish or abort that operation before applying a patch."
            )

    changes = text(git(
        repo, "status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none",
    ).stdout)
    if changes:
        raise PatchError(
            "There are staged or modified tracked files. Commit or stash your work first.\n"
            + changes
        )


def check_patch(repo: Path, contents: bytes) -> None:
    # --check makes this read-only, even with --index. The index-aware check
    # rejects an untracked file masquerading as the original of a modified file.
    result = git(
        repo, "apply", "--check", "--index", "--whitespace=warn", "-",
        patch=contents, check=False,
    )
    if result.returncode:
        detail = text(result.stderr or result.stdout)
        raise PatchError(
            "Patch check failed; no patch changes were applied.\n"
            f"{detail}\n"
            "The patch may already be applied, target a different revision, or collide with a file."
        )
    if result.stderr:
        print(text(result.stderr), file=sys.stderr)


def branch_exists(repo: Path, name: str) -> bool:
    result = git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{name}", check=False)
    if result.returncode not in (0, 1):
        raise PatchError(text(result.stderr) or "Could not check existing branches.")
    return result.returncode == 0


def check_existing_patch_branch(repo: Path) -> bool:
    exists = branch_exists(repo, "patch")
    if exists:
        result = git(repo, "merge-base", "--is-ancestor", "patch", "main", check=False)
        if result.returncode == 1:
            raise PatchError(
                "Branch 'patch' contains commits that are not merged into main.\n"
                "Review and merge that work before applying another patch. "
                "The existing branch has been preserved."
            )
        if result.returncode:
            raise PatchError(text(result.stderr) or "Could not compare patch with main.")
    return exists


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply and commit a patch on branch 'patch', then switch back to 'main'.",
        epilog="Run inside the target repository. Relative patch paths use your current directory.",
    )
    parser.add_argument("patch", type=Path, help="Path to a .patch or unified .diff file")
    parser.add_argument("--check", action="store_true", help="Check and show the patch summary without applying it")
    args = parser.parse_args()

    try:
        patch_path = args.patch.expanduser().resolve(strict=True)
        if not patch_path.is_file():
            raise PatchError(f"Patch path is not a regular file: {patch_path}")
        # Freeze the input so preflight and application use exactly the same bytes,
        # even if the patch modifies its own source file or this running helper.
        contents = patch_path.read_bytes()
        if not contents.strip():
            raise PatchError(f"Patch file is empty: {patch_path}")

        repo = repository_root()
        ensure_ready(repo)
        if not branch_exists(repo, "main"):
            raise PatchError("A local 'main' branch is required. No branches were changed.")
        current = text(git(repo, "branch", "--show-current").stdout)
        if args.check and current != "main":
            raise PatchError("Switch to main before using --check: git switch main")
        exists = False
        if not args.check:
            exists = check_existing_patch_branch(repo)
            # Fail before changing branches if Git lacks a usable author identity.
            git(repo, "var", "GIT_AUTHOR_IDENT")
            git(repo, "var", "GIT_COMMITTER_IDENT")
            if current != "main":
                git(repo, "switch", "main")
                print("Switched to main.", flush=True)
            ensure_ready(repo)
        print(f"Repository: {repo}", flush=True)
        print(f"Patch: {patch_path}", flush=True)
        check_patch(repo, contents)
        summary = git(repo, "apply", "--stat", "-", patch=contents)
        print(text(summary.stdout), flush=True)

        if args.check:
            print("Check passed. No branch or files were changed.")
            return 0

        # Recheck before switching in case the working tree changed during preflight.
        ensure_ready(repo)
        if exists:
            git(repo, "switch", "patch")
            git(repo, "merge", "--ff-only", "main")
            print("Reusing merged branch: patch", flush=True)
        else:
            git(repo, "switch", "-c", "patch", "main")
            print("Created branch: patch", flush=True)
        try:
            ensure_ready(repo)
            result = git(repo, "apply", "--index", "--whitespace=warn", "-", patch=contents)
            if result.stderr:
                print(text(result.stderr), file=sys.stderr)
            # --index stages only the paths changed by this patch. Never use git add .
            committed = git(repo, "commit", "-m", f"Apply patch: {patch_path.name}")
            print(text(committed.stdout), flush=True)
            if committed.stderr:
                print(text(committed.stderr), file=sys.stderr)
            commit_id = text(git(repo, "rev-parse", "--short", "HEAD").stdout)
            # Hooks can change tracked files; do not carry such edits onto main.
            ensure_ready(repo)
            git(repo, "switch", "main")
        except (PatchError, OSError) as exc:
            raise PatchError(
                f"Could not complete the patch workflow: {exc}\n"
                "Your current state has been preserved. Run git status and "
                "git branch --show-current before retrying."
            ) from exc
        print(f"\nCommitted {commit_id} on patch. You are back on main.")
        print("Review changes: git diff main..patch")
        print("To test the revised code: git switch patch")
        print("When ready, merge from main: git merge --ff-only patch")
        return 0
    except (PatchError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Inspect git status and git branch --show-current before retrying.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
