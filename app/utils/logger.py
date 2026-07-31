import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
        ))
        logger.addHandler(handler)
    return logger
