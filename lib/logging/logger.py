import datetime
import logging
import os
import sys
import colorlog


def setup_logger(name, level='INFO'):
    """Function setup as many loggers as you want"""

    level = logging.getLevelName(level)
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the directory for log files
    log_dir = "./logs"
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    # Use the timestamp to create a unique log file for each session
    log_file = f"{log_dir}/log_{timestamp}.log"
    # Create a handler for writing to the log file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')    
    file_formatter = logging.Formatter('%(asctime)s|(%(filename)s)[%(levelname)s]:%(message)s')
    file_handler.setFormatter(file_formatter)

    # Create a handler for writing to the console
    console_handler = logging.StreamHandler()
    console_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
    # Define a color scheme for the log levels
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    console_formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s|[%(levelname)s]%(reset)s:%(message)s',
        datefmt='%H:%M:%S',
        log_colors=log_colors)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger