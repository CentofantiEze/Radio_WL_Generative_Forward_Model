"""Logger configuration for the radio weak-lensing sampling pipeline.

`setup_logger(out_dir)` returns a configured ``logging.Logger`` named
``"shearest"`` with two handlers:

- **stdout**: bare ``"%(message)s"`` — matches the original ``print(...)`` UX so
  SLURM stdout and interactive sessions look the same as before.
- **file**: ``"<out_dir>/radio_sampling.log"`` with timestamps and severity,
  ``"%(asctime)s [%(levelname)s] %(message)s"`` — much more useful for
  postmortem debugging than the previous timestamp-less duplicate of stdout.

The function is idempotent: calling it twice with the same logger name clears
the existing handlers first, which is convenient when running pipelines back
to back in a notebook or test.
"""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FILE_NAME = "radio_sampling.log"
"""Filename used inside ``out_dir`` for the run-level log."""


def setup_logger(
    out_dir: str | Path,
    name: str = "shearest",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the pipeline logger.

    Parameters
    ----------
    out_dir
        Directory where the log file is created. Created if it does not exist.
    name
        Logger name. Defaults to ``"shearest"``.
    level
        Severity threshold for both handlers. Defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger. Use ``logger.info(...)``, ``logger.warning(...)``,
        ``logger.error(...)`` as needed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Idempotency: drop existing handlers so we don't accumulate them when
    # the pipeline is re-run inside the same Python process (notebooks, tests).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    # Don't propagate to the root logger — avoids duplicate output if the
    # caller (notebook / pytest / lib code) has already configured logging.
    logger.propagate = False

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(out_dir / LOG_FILE_NAME, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
