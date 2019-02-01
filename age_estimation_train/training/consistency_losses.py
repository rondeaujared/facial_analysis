import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import SmoothL1Loss, KLDivLoss, NLLLoss

import settings


def get_current_consistency_weight(epoch, consistency_rampup=1):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if consistency_rampup == 0:
        return 1.0
    else:
        current = np.clip(epoch, 0.0, consistency_rampup)
        phase = 1.0 - current / consistency_rampup
        return float(np.exp(-5.0 * phase * phase))


def child_adult_loss(model_out, gt):
    padult = model_out[-1]
    lf = NLLLoss(reduction='mean')
    return lf(padult, gt)


def child_adience_ldl_loss(model_out, gt):
    dist, minor, adience = model_out
    ldl, minor_gt, adience_gt = gt
    lf = NLLLoss(reduction='mean')
    kl = KLDivLoss(reduction='batchmean')

    return kl(dist, ldl) + lf(minor, minor_gt) + lf(adience, adience_gt)


def child_adience_loss(model_out, gt):
    _, minor, adience = model_out
    minor_gt, adience_gt = gt
    lf = NLLLoss(reduction='mean')
    return lf(minor, minor_gt) + lf(adience, adience_gt)


def mean_ldl_loss(model_out, gt):
    dist, age = gt

    kl = KLDivLoss(reduction='batchmean')
    sml = SmoothL1Loss(reduction='mean')

    ages = torch.matmul(torch.exp(model_out), settings.CLASSES).view(-1)

    return kl(model_out, dist) + sml(ages, age)


def stacked_ldl_loss(model_out, epoch, consistency_rampup):

    base_vid_loss = KLDivLoss(reduction='mean').to(settings.DEVICE)

    with torch.no_grad():
        video_ages = torch.matmul(torch.exp(model_out), settings.CLASSES).view(-1)
        means = torch.tensor(list(map(torch.mean, video_ages.split(settings.FRAMES_PER_VID))))
        stds  = torch.tensor(list(map(torch.std, video_ages.split(settings.FRAMES_PER_VID))))
        norms = Normal(means, stds)
        a = torch.tensor(range(0, 101)).reshape(-1, 1).to(torch.float)
        labels = torch.exp(norms.log_prob(a)).t().to(settings.DEVICE)
        target = labels.repeat(settings.FRAMES_PER_VID, 1)

    w = get_current_consistency_weight(epoch, consistency_rampup=consistency_rampup)
    return base_vid_loss(model_out, target) * w


def stacked_mean_loss(model_out, epoch, consistency_rampup):
    base_vid_loss = SmoothL1Loss(reduction='mean')
    video_ages = torch.matmul(torch.exp(model_out), settings.CLASSES).view(-1)
    with torch.no_grad():
        means = torch.tensor(list(map(torch.mean, video_ages.split(settings.FRAMES_PER_VID))))
        target = torch.cat(list(map(lambda x: x.repeat(settings.FRAMES_PER_VID), means))).to(
            dtype=torch.float32, device=settings.DEVICE)
        idx = torch.abs(video_ages-target) < 8
        target[idx] = video_ages[idx]

    w = get_current_consistency_weight(epoch, consistency_rampup=consistency_rampup)
    return base_vid_loss(video_ages, target) * w
