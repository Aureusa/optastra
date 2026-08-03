from __future__ import annotations

import logging
from pathlib import Path


_COLOR_MAP = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLOR_MAP.get(record.levelno)
        prefix = f"[{self.formatTime(record, '%m/%d %H:%M:%S')} | {record.name} {record.levelname}]"
        message = record.getMessage()

        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if color:
            prefix = f"{color}{prefix}{_RESET}"

        return f"{prefix}: {message}"


def setup_logging(
    output_dir: str | Path,
    *,
    filename: str = "optastra.log",
    logger_name: str = "optastra",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    color: bool = True,
) -> logging.Logger:
    """Configure the shared optastra logger tree for console and file output.

    Child loggers like ``optastra.train`` and ``optastra.eval`` inherit this setup.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers so repeated calls stay idempotent.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s %(levelname)s]: %(message)s",
        "%m/%d %H:%M:%S",
    )
    color_formatter = ColorFormatter()

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(color_formatter if color else formatter)
        logger.addHandler(console_handler)

    if file:
        file_handler = logging.FileHandler(output_path / filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger