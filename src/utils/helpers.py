import logging
import os
import time

from src.core import constants as const

LOG_DIR = const.LOG_DIR_PATH
COMMON_LOGS = const.COMMON_LOGS_FILE_PATH


def construct_script_name(file_path: str) -> str:
    """In case you forgot how this works: pass << __file__ >> as the file path
    argument from the script calling this function !"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return base_name


def setup_logger(script_name: str, level: int = logging.DEBUG, log_dir: str = LOG_DIR):
    script_log_file_path = os.path.join(log_dir, f"{script_name}.log")
    common_log_file_path = COMMON_LOGS

    with open(script_log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write("<< New Log Entry >>\n")

    logger = logging.getLogger(script_name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        # Script-specific file handler with UTF-8 encoding
        script_fh = logging.FileHandler(script_log_file_path, encoding="utf-8")
        script_fh.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        script_fh.setFormatter(formatter)
        logger.addHandler(script_fh)

        # Common file handler with UTF-8 encoding
        common_fh = logging.FileHandler(common_log_file_path, encoding="utf-8")
        common_fh.setLevel(level)
        common_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s %(message)s"
        )
        common_fh.setFormatter(common_formatter)
        logger.addHandler(common_fh)

    return logger


def log_empty_lines(logger, lines: int = 1):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.stream.write(lines * "\n")
            handler.flush()


def print_countdown(duration: int = 3):
    for seconds in reversed(range(1, duration)):
        print("\r" + f"Courting down from {seconds} seconds...", end="\r")
        time.sleep(1)


def main():
    script_name = construct_script_name(__file__)
    setup_logger(script_name, 2)


if __name__ == "__main__":
    main()
