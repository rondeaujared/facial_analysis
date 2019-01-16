import logging

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

import settings
from ..datasets.utils import norm_labeler

# @TODO: Look into video logging for Tensorboard
"""
vid_images = dataset.train_data[:16 * 48]
vid = vid_images.view(16, 1, 48, 28, 28)  # BxCxTxHxW

writer.add_video('video', vid_tensor=vid)
writer.add_video('video_1_fps', vid_tensor=vid, fps=1)
"""


class DummyLogger:
    """
        Dummy Object which will do nothing no matter what method is called upon it.
        Pass this if you don't want to do any tensorboard logging
    """

    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


def appa_real_log_figs(outputs, pred, labels, writer, fold, epoch=None):
    logger = logging.getLogger(settings.LOG_NAME)
    dists = torch.exp(outputs.data).clone().cpu().data.numpy()
    y_hat = pred.clone().cpu().data.numpy().astype(int)

    paths = labels['path']
    app_ages = labels['app_age'].clone().cpu().data.numpy().astype(int)
    real_ages = labels['real_age'].clone().cpu().data.numpy().astype(int)
    gt = labels['label'].clone().cpu().data.numpy()

    logger.info(f"{'Image xxx.jpg':>23} (Pr, Ap, Bi) | MAE")
    for p, d, y, aa, ra, t in zip(paths, dists, y_hat, app_ages, real_ages, gt):
        fname = p[p.find('release')+len('release')+1:]
        mae = y-aa
        if abs(mae) > 10:
            level = logger.warning
        else:
            level = logger.info

        level(f"{fname} ({y:2d}, {aa:2d}, {ra:2d}) | {mae:2d}")
        fig = overlay_image(p, d, ['Pred age: ' + str(y),
                                   'App age: ' + str(aa),
                                   'Real age: ' + str(ra)],
                            rdist=t)
        writer.add_figure(str(fold) + str(epoch), fig, epoch)


def umd_faces_log_figs(out_frames, video_ages, paths, writer, epoch=None):
    logger = logging.getLogger(settings.LOG_NAME)
    ages = video_ages.cpu().numpy().astype(int)
    mean = ages.mean()
    var = ages.var()
    logger.info(f"Vid (Ages, Mean, Var): {ages}, {mean}, {var}")
    writer.add_scalars(f"val/vid", {'mean': mean, 'var': var}, epoch)
    return
    dists = torch.exp(out_frames.data).clone().cpu().data.numpy()
    y_hat = video_ages.clone().cpu().data.numpy().astype(int)
    classes = torch.cuda.FloatTensor([range(0, 101)]).t()
    avg = torch.mean(torch.exp(out_frames.data), dim=0)

    # rdist = avg.clone().cpu().data.numpy()

    label = norm_labeler(y_hat)

    avg_age = torch.matmul(avg.data, classes).view(-1).clone().cpu().data.numpy()
    video_name = paths[0]
    video_name = video_name[video_name.rfind('DB')+3:]
    mu = np.mean(y_hat).astype(int)
    in_loss = np.zeros_like(y_hat)
    idx = np.abs(y_hat - mu) > 5
    in_loss[idx] = y_hat[idx]
    in_loss_str = ' '.join([f'{str(x):<2}' for x in in_loss if x > 0])
    y_hat_str = ' '.join([f'{str(x):<2}' for x in y_hat])
    var = np.var(y_hat).astype(int)
    logger.info(f"Ages pred for "
                f"{video_name.replace('aligned_detect', ''):<35}: "
                f"{y_hat_str}"
                f"\t(Mean, Var): ({mu:2d}, "
                f"{var:2d})\t|| "
                f"in loss: {in_loss_str}")
    writer.add_scalars(f"val/vid", {'mean': mu, 'var': var}, epoch)

    for p, d, y in random.sample(list(zip(paths, dists, y_hat)), 5):

        fig = overlay_image(p, d, ['Pred age: ' + str(y),
                                   'Vid avg age: ' + str(avg_age.astype(int)[0])],
                            rdist=label)
        writer.add_figure(str(epoch) + '/' + video_name, fig, epoch)


#  See @https://stackoverflow.com/a/44985962
def overlay_image(path, pred, text=[], rdist=None):
    plt.switch_backend('agg')
    fig = plt.figure()

    # remove upper corner plot
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    ax1.axis('off')

    # plot some example image
    img = mpimg.imread(path)
    img_height, img_width = img.shape[0], img.shape[1]
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax2.imshow(img)
    ax2.axis('off')

    #ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) #, sharey=ax2)
    if rdist is not None:
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)  # , sharey=ax2)
        ax3.barh(range(rdist.shape[0]), rdist, height=1, align='center', color='red')
        for i, t in enumerate(text):
            ax3.text(0, int(100 - 10 * i), t)
    else:
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=ax2)
        ax3.set_yticks([])
        for i, t in enumerate(text):
            ax3.text(0, int(0 + img_height*0.1*i), t)

    ax3.set_frame_on(False)
    ax3.set_xticks([])


    # Histogram over top of image
    bins = pred.shape
    ax4 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  #, sharex=ax2)
    #ax4.bar(np.linspace(0, img.shape[1], bins[0]), pred, width=1, align='center')
    if rdist is not None:
        ax4.bar(range(rdist.shape[0]), rdist, width=1, align='center', color='green')
    ax4.bar(range(bins[0]), pred, width=1, align='center')
    ax4.set_frame_on(False)
    ax4.set_yticks([])

    #fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    return fig
