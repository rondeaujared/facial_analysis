from collections import OrderedDict

import torch.nn.functional as func
from torch import nn
from torch.nn import Linear

from .AgeNet import AgeNet


class AuxAgeNet(AgeNet):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, drop_rate=0, num_classes=101, imagenet=False,
                 freeze_features=False, base_weights='', aux_weights='', **kwargs):

        super().__init__(drop_rate=drop_rate,
                         imagenet=imagenet, freeze_features=freeze_features, base_weights=base_weights,
                         **kwargs)

        self.child_adult = nn.Sequential(OrderedDict([
            ('fc0', Linear(self.classifier.in_features, 2)),
            #('relu0', nn.ReLU(inplace=True)),
            #('fc1', Linear(512, 512)),
            #('relu1', nn.ReLU(inplace=True)),
            #('fc2', Linear(512, 2))
        ]))

        self.adience = nn.Sequential(OrderedDict([
            ('fc0', Linear(self.classifier.in_features, 8)),
            #('relu0', nn.ReLU(inplace=True)),
            #('fc1', Linear(512, 512)),
            #('relu1', nn.ReLU(inplace=True)),
            #('fc2', Linear(512, 8))
        ]))

        if aux_weights:
            self.load_weights(aux_weights)

        for param in self.features.parameters():
            param.requires_grad = not freeze_features

        for param in self.classifier.parameters():
            param.requires_grad = not freeze_features

        for param in self.child_adult.parameters():
            param.requires_grad = True

        for param in self.adience.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        out = func.relu(features, inplace=True)
        out = func.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)

        # Get 101-dim label distribution
        dist = self.classifier(out)
        dist = func.log_softmax(dist, dim=1)

        # minor[0] = p(child), minor[1] = p(adult)
        minor = self.child_adult(out)
        minor = func.log_softmax(minor, dim=1)

        # 8-dims for Adience style grouping
        # Traditionally (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)
        adience = self.adience(out)
        adience = func.log_softmax(adience, dim=1)

        return dist, minor, adience
