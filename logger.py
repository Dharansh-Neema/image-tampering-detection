import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
os.makedirs("logs",exist_ok=True)
def setup_logger(
    name:str = "imgae_tampering_detection",
    log_file:str = "logs/src.log",
    console_level:str="DEBUG",
    file_level:str="DEBUG",
    max_bytes: int = 5*1024*1024, # 5MB
    backup_count: int=3
)->logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_format)


    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    