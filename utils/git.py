from __future__ import annotations

import subprocess
from datetime import datetime
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

    commit_message = f"Updates {datetime.now():%Y-%m-%d %H:%M}"

    run([
        "git",
        "commit",
        "-m",
        commit_message,
    ])


if __name__ == "__main__":
    main()