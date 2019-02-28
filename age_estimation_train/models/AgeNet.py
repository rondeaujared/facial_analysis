import torch
import logging
import torch.nn.functional as func

from re import compile
from torch.nn import Linear
from torch.utils.model_zoo import load_url
from . import DenseNet
import settings


class AgeNet(DenseNet.DenseNet):
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
                 freeze_features=False, base_weights='', **kwargs):
        # DenseNet-121
        super().__init__(drop_rate=drop_rate, num_classes=1000,
                         num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 24, 16), **kwargs)

        # DenseNet-161
        #super().__init__(drop_rate=drop_rate, num_classes=1000,
        #                 num_init_features=96, growth_rate=48,
        #                 block_config=(6, 12, 36, 24), **kwargs)
        self.logger = logging.getLogger(settings.LOG_NAME)

        if imagenet:
            self.logger.info(f"Initializing _WEIGHTS from ImageNet...")
            imagenet = self.get_imagenet()
            self.load_state_dict(imagenet)

        if freeze_features:
            self.logger.info(f"Freezing feature layers")
            for param in self.features.parameters():
                param.requires_grad = False

        self.logger.info(f"Randomly initializing {num_classes} output neurons")
        self.classifier = Linear(self.classifier.in_features, num_classes)

        if base_weights:
            self.load_weights(base_weights)

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        out = func.relu(features, inplace=True)
        out = func.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        out = func.softplus(out)
        # out = func.relu(out, inplace=False)
        # out = func.log_softmax(out, dim=1)
        return out
    
    @staticmethod
    def get_imagenet():
        r"""
        This method fixes backwards compatability issues with state_dicts holding certain
        characters
        :return: state_dict for DenseNet161 containing ImageNet Weights compatible for PyTorch 0.4+
        """
        exp = r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        pattern = compile(exp)
        # state_dict = load_url('https://download.pytorch.org/models/densenet161-8d451a50.pth')
        state_dict = load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return state_dict

    def load_weights(self, path):
        self.logger.info(f"Loading _WEIGHTS from: {path}")
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
            self.logger.info(f"Loaded ijcnn style _WEIGHTS")
        except:
            for key in list(state_dict.keys()):
                new_key = key[:key.find('module')] + key[key.find('module') + len('module') + 1:]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            self.load_state_dict(state_dict)
            self.logger.info(f"Loaded new style _WEIGHTS!")
