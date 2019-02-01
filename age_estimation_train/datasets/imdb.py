from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.io
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

from settings import *
from . import utils
from ..training.preprocessing import ImdbTransformer

MAX_AGE = 80
MIN_AGE = 0
BASE_DIR = '/mnt/fastdata/datasets/imdb/'
FNAME = BASE_DIR + 'imdb.mat'
LOG_NAME = 'age_train'


class ImdbDataset(data.Dataset):

    def __init__(self, transform, root=BASE_DIR, target_transform=None,
                 labeler=utils.imdb_labeler, keep=lambda x: True, n=None):
        if n == 0:
            return

        self.root = root
        self.transform = ImdbTransformer(transform)
        self.target_transform = target_transform
        self.loader = default_loader
        self.logger = logging.getLogger(LOG_NAME)

        imdb = build_imdb(n)
        images = []
        faces = []
        atg = utils.get_age_to_group()
        c_child = 0
        c_adult = 0

        # Pandas(Index=59332, face_location=array([ 73,  92, 237, 256]), face_score=1.2,
        # full_path='54/nm0000454_rm1438423040_1936-5-17_2008.jpg', gender=1.0, photo_taken=2008,
        # celeb_names='Dennis Hopper', bday=Timestamp('1936-05-17 00:00:00'),
        # date_taken=Timestamp('2008-01-01 00:00:00'), age=71., dx=164, dy=164, weight=495)
        i = 0
        for row in imdb.itertuples():
            path = self.root + row[3]
            age = int(round(float(row[-4])))
            group = atg[int(round(float(age)))]
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
                face = {'x1': row[1][0], 'y1': row[1][1], 'x2': row[1][2], 'y2': row[1][3]}
                faces.append(face)
                if target['adult']:
                    c_adult += 1
                else:
                    c_child += 1
            else:
                self.logger.debug(f"{target} was filtered")
            i += 1

        self.logger.info(f"Loaded {(c_child+c_adult)} images from IMDB; {c_child} children and {c_adult} adults.")
        self.images = images
        self.faces = faces

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.images[index]

        img = cv2.imread(path)
        if img is None:
            self.logger.warning(f"Image opened as none! {path}")
        img = utils.image_resize(img)
        if img is None or img.shape[0] > 4000 or img.shape[1] > 4000:
            self.logger.warning(f"Image too large: {img.shape}; ignoring")
            return np.zeros((1, 5))

        try:
            img = img - np.array([104, 117, 123])
            p = np.random.rand()
            if p > 0.50:
                img = cv2.flip(img, flipCode=1)
            img = img.transpose(2, 0, 1)
            img = torch.as_tensor(img, dtype=torch.float)
        except Exception as e:
            self.logger.warning(f"Failed to detect image {path}; with error {e}.")

        '''
        face = self.faces[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img, face)
        '''
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def build_imdb(n=None):
    mat = scipy.io.loadmat(FNAME)
    data = mat['imdb']
    ndata = {n: data[n][0, 0][0] for n in data.dtype.names}

    celeb_names = ndata['celeb_names']
    cn = np.apply_along_axis(lambda x: x[0], 1, celeb_names.reshape(-1, 1))
    cols = {k: ndata[k] for k in ndata.keys() if k != 'celeb_names'}

    df = pd.DataFrame(cols)
    df['celeb_names'] = cn[df['celeb_id']-1]
    origin = np.datetime64('0000-01-01', 'D') - np.timedelta64(1, 'D')
    date = np.array(df['dob']) * np.timedelta64(1, 'D') + origin
    df['bday'] = pd.to_datetime(date, errors='coerce')
    df['date_taken'] = pd.to_datetime(df['photo_taken'], yearfirst=True, format='%Y')
    df['age'] = (df['date_taken'] - df['bday']) / timedelta(days=365)

    # Get rows with exactly 1 face
    df = df[df['second_face_score'].isna()]
    df = df[df['face_score'] > 0]
    df = df[(df.age >= MIN_AGE) & (df.age <= MAX_AGE)]
    df = df.drop(columns=['celeb_id', 'dob', 'name', 'second_face_score'])
    df['full_path'] = df.full_path.apply(lambda x: x[0]).astype(str)

    # In x1,y1,x2,y2
    df['face_location'] = df.face_location.apply(lambda x: x[0].astype(int))
    df['dx'] = df['face_location'].apply(lambda x: x[2]-x[0])
    df['dy'] = df['face_location'].apply(lambda x: x[3]-x[1])
    df = df[df['face_score'] > 2]
    ages = df['age'].apply(lambda x: round(x)).astype(int)
    count = {i: len(ages[ages == i]) for i in range(MIN_AGE, MAX_AGE+1)}
    weights = {i: 1/(count.get(i, 0)/len(ages)) for i in range(MIN_AGE,MAX_AGE+1)}
    df['weight'] = ages.apply(lambda x: weights[x])

    if n is None:
        return df[['face_location', 'face_score',
                   'full_path', 'gender', 'photo_taken', 'celeb_names', 'bday',
                   'date_taken', 'age', 'dx', 'dy', 'weight']]
    else:
        return df.sample(n=n, weights=df.weight)[['face_location', 'face_score',
                                              'full_path', 'gender', 'photo_taken', 'celeb_names', 'bday',
                                              'date_taken', 'age', 'dx', 'dy', 'weight']]
