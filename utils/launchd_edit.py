from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    ROOT / ".env"
)

LAUNCHD_LABEL = os.environ[
    "LAUNCHD_LABEL"
]

PLIST_PATH = (
    Path.home()
    / "Library"
    / "LaunchAgents"
    / f"{LAUNCHD_LABEL}.plist"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not PLIST_PATH.exists():
        raise RuntimeError(
            f"LaunchAgent plist does not exist: "
            f"{PLIST_PATH}"
        )

    editor = os.getenv(
        "EDITOR",
        "nano",
    )

    command = [
        *shlex.split(
            editor
        ),
        str(
            PLIST_PATH
        ),
    ]

    print(
        f"Opening:\n{PLIST_PATH}\n"
    )

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()