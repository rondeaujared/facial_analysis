import torch
import cv2
import os
import types
import inspect
import pickle
import random
import logging
from age_estimation_train.training.consistency_losses import *
from age_estimation_train.models import AuxilliaryAgeNet, AgeNet
from face_detection.sfd.models import s3fd_features
from PIL import ImageFile

cv2.setNumThreads(20)
torch.backends.cudnn.bencmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = os.getcwd() + '/datasets/'

#MODEL = AgeNet.AgeNet
MODEL = s3fd_features.s3fd_features
S3FD_RESIZE = 256
IMAGE_LOSS = child_adult_loss  #gaussian_kl_divergence

CLASSES = torch.cuda.FloatTensor([range(0, 101)]).t()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INITIAL_EVAL = False
SAVE_MODEL = True
REPICKLE = True
DEBUG = False

LOG_NAME = 'age_train'
EXPERIMENT_NAME = 's3fd_features'
OPTIM = 'adam'
WEIGHTS = {
    'none': '',
    's3fd': 'models/s3fd/s3fd_convert.pth',
    's3fd_aux': 'models/s3fd_aux/debugging.pth',
    'exp': 'runs/Mar03_01-07-02_s3fd_features/s3fd_features.pth',
    's3fd_new': 'models/Mar02_16-10-21_s3fd_features/s3fd_features.pth'
}

FRAMES_PER_VID = 8
EPOCHS = 1
BATCH_SIZE = 128

ADAM_PARAMS = {'lr': 0, 'weight_decay': 0, 'betas': (0.9, 0.999)}
SGD_PARAMS  = {'lr': 1e-3, 'weight_decay': 1e-5, 'momentum': 0.9, 'nesterov': True}

MODEL_PARAMS = {
    #'imagenet': True,
    #'freeze_features': False,
    'base_weights': WEIGHTS['s3fd'],
    'aux_weights': WEIGHTS['exp'],
    'drop_rate': 0,
    #'num_classes': 2,
}

SCHEDULER_PARAMS = {'factor': 0.50, 'patience': 10, 'threshold': 1e-2, 'verbose': True}
DATASET_PARAMS = {
    'train': {'appa_real': 0, 'adience': 128, 'imdb': 0, 'dir': 1,
              'root_dir': [
                          #'/mnt/fastdata/datasets/age-crawler/organized_google2/child/',
                          #'datasets/labelled_flickr/',
                          #'datasets/challenging-binary-age/',
                          ],
              # 'txt': [],
              'txt': []  #['datasets/imdb/imdb/imdb_adults.txt',
                         #'datasets/imdb/imdb/imdb_children.txt']
              },

    'val': {'appa_real': 0, 'adience': 0, 'imdb': 0, 'ptrain': 1, 'dir': 1,
            #'root_dir': ['/mnt/data/playground/YouTubeFaces/YouTubeFaces/frame_images_DB/']
            #'root_dir': ['/mnt/data/playground/redlight/images/train/nsfw/']
            'root_dir': ['datasets/nsfw/redlight-images/']
            #'root_dir': ['datasets/challenging-binary-age/']
            #'root_dir': ['datasets/labelled_flickr/']
            },

    'test': {'appa_real': 0, 'adience': 128, 'imdb': 0, 'dir': 0,
             'root_dir': ['/mnt/fastdata/datasets/age-crawler/labelled_flickr/']},
    'video': 0
}

VIDEO_LOSS = lambda x, y: stacked_mean_loss(x, y, 10)
USE_VID_LOSS = lambda step, epoch: False
KEEP = lambda x: True  # Will include item in dataset if evaluates to true; label passed as x
