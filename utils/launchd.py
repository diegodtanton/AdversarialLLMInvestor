"""
Launchd utility for this project's scheduled pipeline.

Options:

    status
        Show whether the LaunchAgent is loaded, its current state,
        PID, last exit code, and configured schedule.

    edit
        Open the LaunchAgent plist in $EDITOR.
        Falls back to nano if $EDITOR is not set.
        Saving changes does NOT automatically reload launchd.

    reload
        Unload and reload the LaunchAgent so changes made to the
        plist take effect. Does not intentionally run the pipeline.

    kickstart
        Run the currently loaded LaunchAgent immediately.
        If it is already running, restart it.
        If it is not loaded, load it first.

    run
        Reload the plist first, then immediately run the LaunchAgent.
        Useful after editing when you want both the new configuration
        applied and the pipeline started now.

Usage:

    python -m utils.launchd status
    python -m utils.launchd edit
    python -m utils.launchd reload
    python -m utils.launchd kickstart
    python -m utils.launchd run
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import subprocess
import sys
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
# LAUNCHCTL
# ============================================================

def launchctl(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess:

    result = subprocess.run(
        [
            "launchctl",
            *args,
        ],
        text=True,
        capture_output=True,
    )

    if (
        check
        and result.returncode != 0
    ):
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or (
                "launchctl "
                + " ".join(args)
                + f" failed with exit code "
                f"{result.returncode}"
            )
        )

        raise RuntimeError(
            message
        )

    return result


def is_loaded() -> bool:

    result = launchctl(
        "print",
        SERVICE,
        check=False,
    )

    return (
        result.returncode
        == 0
    )


# ============================================================
# PLIST
# ============================================================

def ensure_plist_exists() -> None:

    if not PLIST_PATH.exists():
        raise RuntimeError(
            "LaunchAgent plist does not exist:\n"
            f"{PLIST_PATH}"
        )


def load_plist() -> dict:

    ensure_plist_exists()

    with PLIST_PATH.open(
        "rb"
    ) as f:
        return plistlib.load(
            f
        )


def format_schedule(
    plist: dict,
) -> str:

    interval = plist.get(
        "StartCalendarInterval"
    )

    if interval is None:
        return "not configured"

    intervals = (
        interval
        if isinstance(
            interval,
            list,
        )
        else [
            interval
        ]
    )

    formatted = []

    for item in intervals:
        if not isinstance(
            item,
            dict,
        ):
            continue

        hour = item.get(
            "Hour"
        )

        minute = item.get(
            "Minute",
            0,
        )

        weekday = item.get(
            "Weekday"
        )

        if hour is not None:
            time_text = (
                f"{int(hour):02d}:"
                f"{int(minute):02d}"
            )
        else:
            time_text = (
                "calendar interval"
            )

        if weekday is not None:
            time_text += (
                f" | weekday={weekday}"
            )

        formatted.append(
            time_text
        )

    if not formatted:
        return str(
            interval
        )

    return ", ".join(
        formatted
    )


# ============================================================
# STATUS
# ============================================================

def extract_launchctl_value(
    output: str,
    key: str,
) -> str | None:

    prefix = (
        f"{key} = "
    )

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith(
            prefix
        ):
            return line.split(
                "=",
                1,
            )[1].strip()

    return None


def print_status() -> None:

    ensure_plist_exists()

    plist = load_plist()

    result = launchctl(
        "print",
        SERVICE,
        check=False,
    )

    print(
        f"Label:       {LAUNCHD_LABEL}"
    )

    print(
        f"Plist:       {PLIST_PATH}"
    )

    print(
        f"Schedule:    {format_schedule(plist)}"
    )

    if result.returncode != 0:
        print(
            "Loaded:      no"
        )

        print(
            "State:       -"
        )

        print(
            "PID:         -"
        )

        print(
            "Last exit:   -"
        )

        return

    output = (
        result.stdout
    )

    state = extract_launchctl_value(
        output,
        "state",
    )

    pid = extract_launchctl_value(
        output,
        "pid",
    )

    last_exit = extract_launchctl_value(
        output,
        "last exit code",
    )

    print(
        "Loaded:      yes"
    )

    print(
        f"State:       {state or 'idle'}"
    )

    print(
        f"PID:         {pid or '-'}"
    )

    print(
        f"Last exit:   {last_exit or '-'}"
    )


# ============================================================
# EDIT
# ============================================================

def edit_plist() -> None:

    ensure_plist_exists()

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

    print(
        "\nEdit complete."
    )

    print(
        "Run:"
    )

    print(
        "    python -m utils.launchd reload"
    )

    print(
        "to apply any plist changes."
    )


# ============================================================
# LOAD / RELOAD
# ============================================================

def bootstrap() -> None:

    ensure_plist_exists()

    launchctl(
        "bootstrap",
        DOMAIN,
        str(
            PLIST_PATH
        ),
    )


def reload_job() -> None:

    ensure_plist_exists()

    print(
        f"Reloading {LAUNCHD_LABEL}..."
    )

    if is_loaded():

        result = launchctl(
            "bootout",
            SERVICE,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Failed to unload existing "
                "LaunchAgent:\n"
                + (
                    result.stderr.strip()
                    or result.stdout.strip()
                )
            )

    bootstrap()

    print(
        "Reload complete."
    )


# ============================================================
# KICKSTART
# ============================================================

def kickstart_job() -> None:

    if not is_loaded():
        print(
            "LaunchAgent is not loaded; "
            "loading it first..."
        )

        bootstrap()

    print(
        f"Kickstarting {LAUNCHD_LABEL}..."
    )

    launchctl(
        "kickstart",
        "-k",
        SERVICE,
    )

    print(
        "Kickstart requested."
    )


# ============================================================
# RUN
# ============================================================

def run_job() -> None:

    reload_job()

    kickstart_job()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Edit and control this project's "
            "launchd LaunchAgent."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "status",
            "edit",
            "reload",
            "kickstart",
            "run",
        ],
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    try:

        if args.command == "status":

            print_status()

            return

        if args.command == "edit":

            edit_plist()

            return

        if args.command == "reload":

            reload_job()

            time.sleep(
                0.5
            )

            print()

            print_status()

            return

        if args.command == "kickstart":

            kickstart_job()

            time.sleep(
                1
            )

            print()

            print_status()

            return

        if args.command == "run":

            run_job()

            time.sleep(
                1
            )

            print()

            print_status()

            return

        raise RuntimeError(
            f"Unhandled command: "
            f"{args.command}"
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()