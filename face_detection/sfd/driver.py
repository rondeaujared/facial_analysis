import torch
torch.backends.cudnn.bencmark = True
import logging
import sys
from .test import find_faces


def setup_custom_logger(name, file_name, level='DEBUG'):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    screen_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    logger.debug(f"Saving logger {name} to {file_name}")
    return logger


if __name__ == "__main__":
    logger = setup_custom_logger("find_faces", "find_faces.log")
    # dir = '/mnt/data/playground/youtube-minors/thumbnails'
    # dir = '/mnt/data/playground/challenging-binary-age'
    dir = '/mnt/fastdata/datasets/appa-real/appa-real-release/train'
    find_faces(dir, logger, display=False)
