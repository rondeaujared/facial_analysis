import numpy as np
import scipy.stats
import torch
import cv2
from sklearn.neighbors.kde import KernelDensity


def _init_norms():
    norms = {}
    for i in range(0, 101):
        label = np.zeros(101)
        mean = i
        std = 1
        norm = scipy.stats.norm(mean, std)

        for j in range(0, 101):
            label[j] = norm.pdf(j)

        s = label.sum()
        label = label / s
        norms[i] = label
    return norms


_NORMS = _init_norms()


def imdb_labeler(age):
    age = int(round(age))
    label = _NORMS[age]
    return label


def norm_labeler(picks):
    if isinstance(picks, torch.Tensor):
        picks = picks.clone().cpu().data.numpy().astype(int)

    label = np.zeros(101)
    mean = picks.mean()
    std = picks.std()
    if std < 0.25:
       std = 0.25
    norm = scipy.stats.norm(mean, std)

    for i in range(0, 101):
        label[i] = norm.pdf(i)

    s = label.sum()
    assert s > 1e-5, "\n".join([str(f"{key} {value}\n")
                                for key, value in
                                {'Sum': s, 'Picks': picks, 'Label': label, 'Mean': mean, 'Std': std}.items()])

    label = label / s
    return label


def kde_labeler(picks):
    if isinstance(picks, torch.Tensor):
        picks = picks.clone().cpu().data.numpy().astype(int)
    nums = np.array([x for x in range(0, 101)]).reshape(-1, 1)
    picks = picks.reshape(-1, 1)
    lower = np.percentile(picks, 25)
    upper = np.percentile(picks, 75)
    IQR = upper - lower
    std = picks.std()
    if std < 0.5:
        std = 1.0
        IQR = 1.0

    if IQR < 0.1:
        IQR = 0.1
    m = min(np.sqrt(std * std), IQR / 1.349)
    bandwidth = (0.9 * float(m)) / (float(pow(float(len(picks)), 0.2)))

    if bandwidth > 5:
        # TODO: Handle this in a manner not using print statements. Maybe set a warning flag
        print(f"Bandwidth too high! m: {m} std: {std} IQR: {IQR} bandwidth: {bandwidth}")

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(picks)

    log_dens = kde.score_samples(nums)
    label = np.exp(log_dens)
    label = label / label.sum()
    return label


def get_age_to_group():
    atg = {}
    # Groups: (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)
    #           1    5    10.5   17.5   28.5   40.5   50.5
    for i in range(0, 3+1):     # 4     [0,3]
        atg[i] = 0
    for i in range(4, 7+1):     # 4     [4,7]
        atg[i] = 1
    for i in range(8, 14+1):    # 7     [8,14]
        atg[i] = 2
    for i in range(15, 22+1):   # 7     [15,22]
        atg[i] = 3
    for i in range(23, 35+1):   # 13    [23,35]
        atg[i] = 4
    for i in range(36, 45+1):   # 10    [36,45]
        atg[i] = 5
    for i in range(46, 56+1):   # 11    [46,56]
        atg[i] = 6
    for i in range(57, 101):    # 44    [57,100]
        atg[i] = 7

    return atg


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
            p = np.random.rand()
            if p > 0.50:
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
    if p > 0.50:
        image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        image = cv2.copyMakeBorder(image, left=wpad, top=hpad, bottom=0, right=0,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image
