import logging
from logging.handlers import RotatingFileHandler
from app.config import LOG_FILE


def setup_logging() -> None:
    """Idempotent RotatingFileHandler + console logging for the app."""
    logger = logging.getLogger("TaskFlowAgent")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return  # already configured

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)