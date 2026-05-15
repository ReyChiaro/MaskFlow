import os
import sys

from loguru import logger


def setup_logger(is_main_process: bool, rank: int, log_dir: str, log_filename: str, log_per_rank: bool):
    logger.remove()
    log_path = os.path.join(log_dir, f"{log_filename}-rank{rank}.log")

    if is_main_process:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True,
        )
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="DEBUG",
            rotation="50 MB",
            enqueue=True,
        )
    elif log_per_rank:
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="DEBUG",
            rotation="50 MB",
            enqueue=True,
        )

    return logger
