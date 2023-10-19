"""Create a custom logger. Used to replace all print calls. Outputs to log file."""
import logging
import sys


def setup_custom_logger(name: str, level=logging.DEBUG, logfile: str = "log.log") -> logging.Logger:
    """Create custom logger. Auto-formats logs lines, writes to file.

    Args:
        logfile: path to log file to write to
        name: logger name
        level: log level (debug/info/..)

    Returns:
        logger
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(name)-20s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(logfile, "a", encoding="utf-8")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for hand in logger.handlers[:]:  # remove logger if already exists (jupyter notebook, ...)
        logger.removeHandler(hand)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    logger.propagate = False
    return logger
