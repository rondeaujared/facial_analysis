from torch.utils.data import ConcatDataset, random_split

from .adience import *
from .appa_real import *
from .imdb import *


def get_train(n, transform):
    todo = []

    t = n.get('appa_real', 0)
    if t is None or t > 0:
        appa_tr = AppaRealDataset(subset='train/', transform=transform, n=t, ext='', crop_faces=True)
        todo.append(appa_tr)

    t = n.get('adience', 0)
    if t is None or t > 0:
        adience_tr = AdienceDataset(transform=transform, subset='train', n=t)
        todo.append(adience_tr)

    t = n.get('imdb', 0)
    if t is None or t > 0:
        imdb_tr = ImdbDataset(transform=transform, n=t)
        todo.append(imdb_tr)
    tr = ConcatDataset(todo)
    return tr


def get_validation(n, transform, to_split=None):
    todo = []

    t = n.get('appa_real', 0)
    if t is None or t > 0:
        appa_val = AppaRealDataset(subset='val/', transform=transform, n=t, ext='', crop_faces=True)
        todo.append(appa_val)

    t = n.get('adience', 0)
    if t is None or t > 0:
        adience_val = AdienceDataset(transform=transform, subset='val', n=t)
        todo.append(adience_val)

    if to_split:
        p = n['ptrain']
        tr, val = random_split(to_split, (int(round(len(to_split)*p)),
                                          int(round(len(to_split)*(1-p)))))
        todo.append(val)
    else:
        tr = None
    val = ConcatDataset(todo)
    return tr, val


def get_test(n, transform):
    todo = []

    t = n.get('appa_real', 0)
    if t is None or t > 0:
        appa_ts = AppaRealDataset(subset='test/', transform=transform, n=t, ext='', crop_faces=True)
        todo.append(appa_ts)

    t = n.get('adience', 0)
    if t is None or t > 0:
        adience_ts = AdienceDataset(transform=transform, subset='test', n=t)
        todo.append(adience_ts)

    ts = ConcatDataset(todo)
    return ts

