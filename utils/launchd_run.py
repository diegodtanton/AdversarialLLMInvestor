from __future__ import annotations

import os
import subprocess
import time
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

DOMAIN = (
    f"gui/{os.getuid()}"
)

SERVICE = (
    f"{DOMAIN}/{LAUNCHD_LABEL}"
)


# ============================================================
# HELPERS
# ============================================================

def launchctl(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess:

    return subprocess.run(
        [
            "launchctl",
            *args,
        ],
        text=True,
        capture_output=True,
        check=check,
    )


def print_status() -> bool:

    result = launchctl(
        "print",
        SERVICE,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"{LAUNCHD_LABEL}: not loaded"
        )

        return False

    output = result.stdout

    pid = None
    state = None
    last_exit_status = None

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith(
            "pid = "
        ):
            pid = line.split(
                "=",
                1,
            )[1].strip()

        elif line.startswith(
            "state = "
        ):
            state = line.split(
                "=",
                1,
            )[1].strip()

        elif line.startswith(
            "last exit code = "
        ):
            last_exit_status = line.split(
                "=",
                1,
            )[1].strip()

    print(
        f"Label:      {LAUNCHD_LABEL}"
    )

    print(
        f"Loaded:     yes"
    )

    print(
        f"State:      {state or 'idle'}"
    )

    print(
        f"PID:        {pid or '-'}"
    )

    print(
        f"Last exit:  {last_exit_status or '-'}"
    )

    return True


def ensure_loaded() -> None:

    result = launchctl(
        "print",
        SERVICE,
        check=False,
    )

    if result.returncode == 0:
        return

    if not PLIST_PATH.exists():
        raise RuntimeError(
            f"Cannot load LaunchAgent because "
            f"plist does not exist:\n{PLIST_PATH}"
        )

    print(
        "\nLaunchAgent is not loaded."
    )

    print(
        f"Loading {PLIST_PATH}..."
    )

    result = launchctl(
        "bootstrap",
        DOMAIN,
        str(
            PLIST_PATH
        ),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "launchctl bootstrap failed:\n"
            + (
                result.stderr.strip()
                or result.stdout.strip()
            )
        )


def kickstart() -> None:

    ensure_loaded()

    print(
        f"\nKickstarting {LAUNCHD_LABEL}..."
    )

    result = launchctl(
        "kickstart",
        "-k",
        SERVICE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "launchctl kickstart failed:\n"
            + (
                result.stderr.strip()
                or result.stdout.strip()
            )
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Current status"
    )

    print(
        "=============="
    )

    print_status()

    kickstart()

    # Give launchd a moment to create the process.
    time.sleep(
        1
    )

    print(
        "\nStatus after kickstart"
    )

    print(
        "======================"
    )

    print_status()


if __name__ == "__main__":
    main()