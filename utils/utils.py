import cv2
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
import os

from constants import LOG_LEVEL


def create_logger(path,backup_count):
    
    # Create a logger object specific to the camera_name
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    full_path = os.path.join(log_directory, path)
    logger = logging.getLogger(full_path)
    if LOG_LEVEL.lower() == "debug":
        logger.setLevel(logging.DEBUG)
    elif LOG_LEVEL.lower() == "info":
        logger.setLevel(logging.INFO)
    elif LOG_LEVEL.lower() == "warning":
        logger.setLevel(logging.WARNING)
    elif LOG_LEVEL.lower() == "error":
        logger.setLevel(logging.ERROR)
    elif LOG_LEVEL.lower() == "critical":
        logger.setLevel(logging.CRITICAL)

    # Ensure no duplicate handlers
    if not logger.hasHandlers():
        # File handler setup
        file_handler = TimedRotatingFileHandler(f"{full_path}.log", when="M", interval=30, backupCount=backup_count, utc=False)
        file_handler.suffix = "%Y-%m-%d_%H-%M"  # Add timestamp to log file

        # File formatter setup (without colors)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)
        
        print("Logger created at path", f"{path}.log")
    else:
        print("Logger already exists for", path)

    return logger


def get_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[:4]
    x3, y3, x4, y4 = bbox2[:4]
    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection / union

def get_waiting_time(seconds):
    """
    convert seconds to MM:SS
    """
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"