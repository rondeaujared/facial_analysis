import logging
import sys
import cv2
import numpy as np
import torch
import glob
import torch.utils.data as data
import os

LOG_NAME = 'find_faces'
FILE_NAME = 'find_faces.log'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EvalDirectory(data.Dataset):

    def __init__(self, root):
        self.root = root
        self.logger = logging.getLogger(LOG_NAME)
        self.images = []
        toglob = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        hits = glob.glob(root + '**', recursive=True)
        self.images = [x for x in hits if x[x.rfind('.'):] in toglob]

    def __getitem__(self, index):
        path = self.images[index]
        img = cv2.imread(path)
        img = image_resize(img)
        if img is None or img.shape[0] > 4000 or img.shape[1] > 4000:
            self.logger.warning(f"Image too large: {img.shape}; ignoring")
            return np.zeros((1, 5))

        try:
            img = img - np.array([104, 117, 123])
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).to(dtype=torch.float)
        except Exception as e:
            self.logger.warning(f"Failed to detect image {path}; with error {e}.")

        return img, path

    def __len__(self):
        return len(self.images)


def image_resize(image, width=None, height=None, big_side=640, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        if max(h, w) < big_side:
            hpad = big_side - h
            wpad = big_side - w
            image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return image
        if h > w:
            height = big_side
        else:
            width = big_side

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    image = cv2.resize(image, dim, interpolation=inter)
    (h, w) = image.shape[:2]
    if width is None: # Height was big side
        wpad = big_side - w
        hpad = 0
    else:
        wpad = 0
        hpad = big_side - h
    image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def setup_custom_logger(name=LOG_NAME, file_name=FILE_NAME, level='DEBUG'):
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


LOGGER = setup_custom_logger()
