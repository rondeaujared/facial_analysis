import torch
import cv2
import os
import types
import inspect
import pickle
import random
import logging
from age_estimation_train.training.consistency_losses import stacked_mean_loss, child_adience_ldl_loss
from age_estimation_train.models import AuxilliaryAgeNet
from PIL import ImageFile

cv2.setNumThreads(0)
torch.backends.cudnn.bencmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL = AuxilliaryAgeNet.AuxAgeNet
IMAGE_LOSS = child_adience_ldl_loss

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
        'none': ''
}

FRAMES_PER_VID = 8
EPOCHS = 5
BATCH_SIZE = 64

ADAM_PARAMS = {'lr': 3e-5, 'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
SGD_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-5, 'momentum': 0.9, 'nesterov': True}
MODEL_PARAMS = {
    'imagenet': True,
    'freeze_features': True,
    'base_weights': WEIGHTS['none'],
    'aux_weights': WEIGHTS['none'],
    'drop_rate': 0.20
}

SCHEDULER_PARAMS = {'factor': 0.50, 'patience': 5, 'threshold': 1e-3, 'verbose': True}
DATASET_PARAMS = {
    'train': {'appa_real': 0, 'adience': None, 'imdb': 0},
    'val': {'appa_real': 0, 'adience': None, 'imdb': 0, 'ptrain': 1},
    'test': {'appa_real': 0, 'adience': None, 'imdb': 0},
    'video': 0
}

VIDEO_LOSS = lambda x, y: stacked_mean_loss(x, y, 10)
USE_VID_LOSS = lambda step, epoch: False
KEEP = lambda x: True  # Will include item in dataset if evaluates to true; label passed as x
