import logging
import sys
import cv2
import numpy as np
import torch
import glob
import torch.utils.data as data
from age_estimation_train.training.preprocessing import S3fdTransformer

import os
import torchvision.transforms as tran

#LOG_NAME = 'find_faces'
LOG_NAME = 'age_train'
FILE_NAME = 'find_faces.log'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EvalDirectory(data.Dataset):

    def __init__(self, root, txt, labels=False, static=False):
        self.logger = logging.getLogger(LOG_NAME)
        self.use_labels = labels
        self.images = []
        self.labels = {}
        self.static = static
        self.transformer = S3fdTransformer(static)

        self.logger.info(f"Root dir {root}")
        for dir in root:
            self.logger.info(f"Loading dir {dir}...")
            self._use_dir(dir)

        for f in txt:
            if 'child' in f:
                self._use_text(f, 0)
            elif 'adult' in f:
                self._use_text(f, 1)

    def _use_dir(self, path, l=None):
        ugh = [f'.~{x}~' for x in range(10)]
        toglob = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'] + ugh
        hits = glob.glob(path + '**', recursive=True)
        images = [x for x in hits if x[x.rfind('.'):] in toglob]
        self.images.extend(images)
        self.logger.info(f"{len(images)} images added from {path}")
        if self.use_labels:
            for img in images:
                label = {
                    'label': np.random.rand((101)),
                    'app_age': float(-1),
                    'path': img,
                    'real_age': float(-1),
                    'group': -1,
                    'adult': -1,
                }

                if l:
                    label['adult'] = l
                elif 'child' in img:
                    label['adult'] = 0
                elif 'adult' in img:
                    label['adult'] = 1
                elif 'nsfw' in img:
                    label['adult'] = 1
                elif 'redlight' in img:
                    label['adult'] = 1
                elif 'minors' in img:
                    label['adult'] = 0
                else:
                    self.logger.warning(f"Labels enabled but no label found for image {img}")
                self.labels[img] = label

    def _use_text(self, path, l):
        f = open(path, "r")
        lines = list(map(lambda x: x.replace('\n', ''), f.readlines()))
        for p in lines:
            label = {
                'label': np.random.rand((101)),
                'app_age': float(-1),
                'path': p,
                'real_age': float(-1),
                'group': -1,
                'adult': -1,
            }

            p = p.replace('imdb', 'imdb/')
            label['path'] = p
            label['adult'] = l

            self.images.append(p)
            self.labels[p] = label
        f.close()

    def __getitem__(self, index):
        path = self.images[index]
        img = self.transformer(path)

        if self.use_labels:
            label = self.labels[path]
            return img, label
        else:
            return img, path

    def __len__(self):
        return len(self.images)


def image_resize(image, width=None, height=None, big_side=640, inter=cv2.INTER_AREA, static=False):
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
            p = np.random.rand()
            if p > 0.50 and not static:
                image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                           borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                image = cv2.copyMakeBorder(image, left=wpad, top=hpad, bottom=0, right=0,
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

    p = np.random.rand()
    if p > 0.50 and not static:
        image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        image = cv2.copyMakeBorder(image, left=wpad, top=hpad, bottom=0, right=0,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
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


#LOGGER = setup_custom_logger()
LOGGER = logging.getLogger(LOG_NAME)