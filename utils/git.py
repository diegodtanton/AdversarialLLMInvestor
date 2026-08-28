from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    run([
        "git",
        "add",
        ".",
    ])

    run([
        "git",
        "commit",
        "-m",
        "Updates",
    ])


if __name__ == "__main__":
    main()