from __future__ import annotations

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = ROOT / "scripts"

OUTPUT_PATH = ROOT / "concat.txt"

MAX_TREE_DEPTH = 3

# Show these directories in the tree, but don't descend into them.
COLLAPSED_DIRS = {
    "data",
    "logs",
}

# Completely hide these.
IGNORED_NAMES = {
    ".git",
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}

# Never concatenate these even if they somehow appear under scripts.
IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
}


# ============================================================
# TREE
# ============================================================

def should_ignore(path: Path) -> bool:
    return (
        path.name in IGNORED_NAMES
        or path.name.startswith(".git")
    )


def sorted_children(directory: Path) -> list[Path]:
    children = [
        path
        for path in directory.iterdir()
        if not should_ignore(path)
    ]

    # Directories first, then files.
    return sorted(
        children,
        key=lambda path: (
            not path.is_dir(),
            path.name.casefold(),
        ),
    )


def build_tree(
    directory: Path,
    *,
    depth: int = 0,
    prefix: str = "",
) -> list[str]:

    if depth >= MAX_TREE_DEPTH:
        return []

    children = sorted_children(directory)

    lines: list[str] = []

    for index, path in enumerate(children):
        is_last = (
            index == len(children) - 1
        )

        branch = (
            "└── "
            if is_last
            else "├── "
        )

        if path.is_dir():
            display_name = (
                f"{path.name}/"
            )
        else:
            display_name = path.name

        lines.append(
            f"{prefix}{branch}{display_name}"
        )

        if not path.is_dir():
            continue

        if path.name in COLLAPSED_DIRS:
            continue

        child_prefix = (
            prefix
            + (
                "    "
                if is_last
                else "│   "
            )
        )

        lines.extend(
            build_tree(
                path,
                depth=depth + 1,
                prefix=child_prefix,
            )
        )

    return lines


def repo_tree() -> str:
    lines = [
        f"{ROOT.name}/",
        *build_tree(ROOT),
    ]

    return "\n".join(lines)


# ============================================================
# SCRIPT CONCATENATION
# ============================================================

def script_files() -> list[Path]:
    files = []

    for path in SCRIPTS_DIR.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in IGNORED_NAMES
            for part in path.parts
        ):
            continue

        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue

        # For this repo, scripts are Python source files.
        if path.suffix.lower() != ".py":
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda path: str(
            path.relative_to(ROOT)
        ).casefold(),
    )


def concatenate_scripts() -> str:
    sections: list[str] = []

    for path in script_files():
        relative_path = path.relative_to(ROOT)

        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            print(
                f"Skipping non-text file: "
                f"{relative_path}"
            )
            continue

        section = "\n".join(
            [
                "=" * 80,
                f"FILE: {relative_path}",
                "=" * 80,
                "",
                content.rstrip(),
                "",
            ]
        )

        sections.append(section)

    return "\n".join(sections)


# ============================================================
# OUTPUT
# ============================================================

def build_output() -> str:
    return "\n".join(
        [
            "=" * 80,
            "REPOSITORY TREE",
            "=" * 80,
            "",
            repo_tree(),
            "",
            "",
            "=" * 80,
            "SCRIPTS",
            "=" * 80,
            "",
            concatenate_scripts(),
            "",
        ]
    )


def main() -> None:
    output = build_output()

    OUTPUT_PATH.write_text(
        output,
        encoding="utf-8",
    )

    files = script_files()

    print(
        f"Wrote {OUTPUT_PATH}"
    )
    print(
        f"Concatenated {len(files)} script files."
    )


if __name__ == "__main__":
    main()