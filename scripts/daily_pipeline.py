from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


STAGES = [
    "news_collect",
    "event_memory",
    "gilman",
]


def run_stage(
    module: str,
) -> None:

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting stage: %s",
        module,
    )

    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            module,
        ],
        cwd=ROOT,
        check=True,
    )

    logger.info(
        "Completed stage: %s",
        module,
    )


def main():
    logger.info(
        "Starting daily news + Gilman pipeline"
    )

    for module in STAGES:
        run_stage(
            module
        )

    logger.info(
        "Daily pipeline complete"
    )


if __name__ == "__main__":
    main()