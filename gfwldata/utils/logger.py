import logging
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path


def setup_logger(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console and optional file output"""
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def setup_multiproc_logger(
    log_file: Path | None = None, level: int = logging.INFO
) -> tuple[mp.Queue, QueueListener]:
    # Configure the main process logger.
    setup_logger(log_file, level)

    # Get a copy of current handlers (e.g., console and file from setup_logger).
    main_handlers = logging.getLogger().handlers[:]

    # Create the log queue.
    log_queue = mp.Queue(-1)

    # Create a QueueListener to forward log items from log_queue to main_handlers.
    listener = QueueListener(log_queue, *main_handlers)

    return log_queue, listener


def init_worker_logger(log_queue: mp.Queue) -> None:
    root_logger = logging.getLogger()

    # Remove any preexisting handlers in the worker process.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Attach the QueueHandler.
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)
