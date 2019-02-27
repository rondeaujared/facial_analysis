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

cv2.setNumThreads(0)
torch.backends.cudnn.bencmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = os.getcwd() + '/datasets/'

MODEL = AgeNet.AgeNet  #s3fd_features.s3fd_features
IMAGE_LOSS = gaussian_kl_divergence

CLASSES = torch.cuda.FloatTensor([range(0, 101)]).t()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INITIAL_EVAL = False
SAVE_MODEL = True
REPICKLE = True
DEBUG = False

LOG_NAME = 'age_train'
EXPERIMENT_NAME = 'debugging'
OPTIM = 'adam'
WEIGHTS = {
    'none': '',
    's3fd': '/home/research/jrondeau/research/risp-prototype/engine/face_detection/sfd/data/s3fd_convert.pth',
    's3fd_aux': '/home/research/jrondeau/research/facial_analysis/runs/Jan26_06-05-31_debugging/debugging.pth'
}

FRAMES_PER_VID = 8
EPOCHS = 200
BATCH_SIZE = 256

ADAM_PARAMS = {'lr': 3e-5,
               'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
SGD_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-5, 'momentum': 0.9, 'nesterov': True}
MODEL_PARAMS = {
    'imagenet': True,
    'freeze_features': False,
    'base_weights': WEIGHTS['none'],
    #'aux_weights': WEIGHTS['s3fd_aux'],
    'drop_rate': 0.20,
    'num_classes': 2,
}

SCHEDULER_PARAMS = {'factor': 0.50, 'patience': 10, 'threshold': 1e-2, 'verbose': True}
DATASET_PARAMS = {
    'train': {'appa_real': 0, 'adience': 0, 'imdb': 100000, 'dir': 0,
              'root_dir': [],
              #            '/mnt/fastdata/datasets/age-crawler/organized_google2/child/',
              #            '/mnt/fastdata/datasets/age-crawler/labelled_flickr/',
              #            ],
              'txt': [],
              #'txt': ['/mnt/fastdata/datasets/imdb/imdb_adults.txt',
              #        '/mnt/fastdata/datasets/imdb/imdb_children.txt']
              },

    'val': {'appa_real': 0, 'adience': 0, 'imdb': 128, 'ptrain': 0.95, 'dir': 0,
            #'root_dir': ['/mnt/data/playground/YouTubeFaces/YouTubeFaces/frame_images_DB/']
            #'root_dir': ['/mnt/data/playground/redlight/images/train/nsfw/']
            #'root_dir': ['/mnt/fastdata/datasets/redlight/redlight-images/SNF/']
            #'root_dir': ['datasets/challenging-binary-age/child/']
            'root_dir': []
            },

    'test': {'appa_real': 0, 'adience': 128, 'imdb': 0, 'dir': 0,
             'root_dir': ['/mnt/fastdata/datasets/age-crawler/labelled_flickr/']},
    'video': 0
}

VIDEO_LOSS = lambda x, y: stacked_mean_loss(x, y, 10)
USE_VID_LOSS = lambda step, epoch: False
KEEP = lambda x: True  # Will include item in dataset if evaluates to true; label passed as x
