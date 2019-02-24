import logging

import numpy as np
import pandas as pd
import scipy.io
import torch.utils.data as data
import cv2
import torch

from torchvision.datasets.folder import default_loader

from . import utils
from ..training.preprocessing import ImdbTransformer

BASE_DIR = 'datasets/appa-real/appa-real-release/'
LOG_NAME = 'age_train'
TRAIN_CSV = 'gt_train.csv'
VAL_CSV = 'gt_valid.csv'
TEST_CSV = 'gt_test.csv'


class AppaRealDataset(data.Dataset):
    r"""
        A CSV data loader for the APPA-REAL dataset, where the rows of the CSV are like this:

        007611.jpg,20,32,54,female
        007612.jpg,47,44,55,female
        007612.jpg,47,40,22,male
        007612.jpg,47,41,33,female

        Columns:
        file_name,real_age,apparent_age,worker_age,worker_gender
    """
    def __init__(self, subset, root=BASE_DIR, csv_path=None, ext='_face.jpg', crop_faces=False,
                 transform=None, target_transform=None, labeler=utils.norm_labeler,
                 writer=None, n=None, keep=lambda x: True):

        valid = ['train/', 'val/', 'test/']
        assert subset in valid, f"Subset must be one of: {' '.join(valid)}"
        assert transform is not None, f"transform is None {transform}"

        if csv_path is None:
            if subset == 'train/':
                csv_path = root + TRAIN_CSV
            elif subset == 'val/':
                csv_path = root + VAL_CSV
            elif subset == 'test/':
                csv_path = root + TEST_CSV

        names = ['file_name', 'real_age', 'apparent_age', 'worker_age', 'worker_gender']

        if crop_faces:
            transform = ImdbTransformer(transform)

        self.root = root+subset
        self.subset = subset[:-1]
        self.csv = pd.read_csv(csv_path, header=0, names=names)
        self.ext = ext
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.logger = logging.getLogger(LOG_NAME)
        self.writer = writer
        self.logger.info(f"APPA-REAL subset {subset} using extension {ext}")
        self.faces = []
        self.crop_faces = crop_faces

        picks = {}
        bio_age = {}
        images = []
        atg = utils.get_age_to_group()
        c_child = 0
        c_adult = 0

        for row in self.csv.itertuples():
            picks[row[1]] = picks.get(row[1], []) + [row[3]]
            bio_age[row[1]] = row[2]

            if n is not None and len(picks.keys()) > n:
                break
        i = 0
        for k, val in picks.items():
            if i % 100 == 0:
                self.logger.debug(f"APPA-REAL: On image {i}")

            val = np.array(val)
            label = labeler(val)
            path = self.root + k + self.ext
            if crop_faces:
                face = getface(self.root + k + '.mat')
            group = atg[int(round(float(bio_age[k])))]
            target = {
                'label': label,
                'app_age': float(val.mean()),
                'path': path,
                'real_age': float(bio_age[k]),
                'group': group,
                'adult': group >= 3,
            }

            if keep(target):
                images.append((path, target))
                if crop_faces:
                    self.faces.append(face)
                i += 1
                if target['adult']:
                    c_adult += 1
                else:
                    c_child += 1
            else:
                self.logger.debug(f"{target} was filtered")

        self.logger.info(f"Loaded {(c_child+c_adult)} images from APPA-REAL; {c_child} children and {c_adult} adults.")
        self.images = images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.images[index]

        if hasattr(self.transform, 'S3FD'):
            img = self.transform(path)
        else:
            img = self.loader(path)
            if self.crop_faces:
                face = self.faces[index]
                img = self.transform(img, face)
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def getface(path):
    mat = scipy.io.loadmat(path)
    t = mat['fileinfo']['face_location'].squeeze().item().squeeze()
    face = {'x1': int(t[0]), 'y1': int(t[1]), 'x2': int(t[2]), 'y2': int(t[3])}
    return face
