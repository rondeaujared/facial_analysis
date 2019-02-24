import pandas as pd
import torch.utils.data as data
import numpy as np
import torch

from numpy.random import randint
from torchvision.datasets.folder import default_loader

from settings import *
from .utils import get_age_to_group, imdb_labeler, image_resize
BASE_DIR = 'datasets/adience/faces/'


class AdienceDataset(data.Dataset):

    def __init__(self, transform, subset, root=BASE_DIR, target_transform=None,
                 labeler=imdb_labeler, keep=lambda x: True, n=None):
        if n == 0:
            return

        valid = ['train', 'val', 'test']
        assert subset in valid, f"Subset must be one of: {' '.join(valid)}"

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.logger = logging.getLogger(LOG_NAME)

        fname = 'age_estimation_train/datasets/data_labels/adience/age_' + subset + '.txt'
        #fname = root + 'age_' + subset + '.txt'
        adience = build_adience(fname, n)
        atg = get_age_to_group()
        c_child = 0
        c_adult = 0

        images = []

        # Pandas(Index, path, group, age)
        i = 0
        for row in adience.itertuples():
            path = self.root + row[1]
            age = int(round(float(row[3])))
            group = int(row[2])

            target = {
                'label': labeler(age),
                'app_age': age,
                'path': path,
                'real_age': age,
                'group': group,
                'adult': group >= 3,
            }

            if keep(target):
                images.append((path, target))
                if target['adult']:
                    c_adult += 1
                else:
                    c_child += 1
            else:
                self.logger.debug(f"{target} was filtered")
            i += 1

        self.logger.info(f"Loaded {(c_child+c_adult)} images from ADIENCE; {c_child} children and {c_adult} adults.")
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
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def build_adience(fname, n=None):
    data = pd.read_csv(fname, sep=' ', names=['path', 'group'])
    data['path'] = data['path'].apply(lambda x: x[:x.find('/') + 1] + 'coarse_tilt_aligned_face.' + x[x.find('.') + 1:])
    # Groups: (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)
    data['group'] = data['group'].astype(int)

    gta = {}
    gta[0] = (0, 2)
    gta[1] = (4, 6)
    gta[2] = (8, 13)
    gta[3] = (15, 20)
    gta[4] = (25, 32)
    gta[5] = (38, 43)
    gta[6] = (48, 53)
    gta[7] = (60, 80)

    data['age'] = data['group'].apply(lambda x: randint(gta[x][0], gta[x][1]+1))
    if n is None:
        return data[['path', 'group', 'age']]
    else:
        return data.sample(n=n)[['path', 'group', 'age']]
