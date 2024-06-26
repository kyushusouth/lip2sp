import logging
from typing import Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages based on their level.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if record.levelno == logging.DEBUG:
            log_fmt = Fore.BLUE + log_fmt + Style.RESET_ALL
        elif record.levelno == logging.INFO:
            log_fmt = Fore.GREEN + log_fmt + Style.RESET_ALL
        elif record.levelno == logging.WARNING:
            log_fmt = Fore.YELLOW + log_fmt + Style.RESET_ALL
        elif record.levelno == logging.ERROR:
            log_fmt = Fore.RED + log_fmt + Style.RESET_ALL
        elif record.levelno == logging.CRITICAL:
            log_fmt = Fore.RED + Style.BRIGHT + log_fmt + Style.RESET_ALL

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up and returns a logger with the specified configuration.

    Args:
        name (str): The name of the logger.
        log_file (Optional[str]): Path to a log file. If None, logs will not be written to a file.
        level (int): The logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create a colored formatter
    formatter = ColoredFormatter()

    # Create a stream handler for output to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    # If log_file is specified, create and add a file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        # Use a non-colored formatter for the file handler
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
