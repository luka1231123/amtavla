import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(log_file: str = "logs/amtavla.log") -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    if getattr(root, "_amtavla_configured", False):
        return

    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    noisy_loggers = {
        "httpcore": logging.WARNING,
        "httpx": logging.INFO,
        "urllib3": logging.WARNING,
        "h2": logging.WARNING,
        "rustls": logging.WARNING,
        "hyper_util": logging.WARNING,
        "cookie_store": logging.WARNING,
        "reqwest": logging.WARNING,
    }
    for name, level in noisy_loggers.items():
        logging.getLogger(name).setLevel(level)

    root._amtavla_configured = True
