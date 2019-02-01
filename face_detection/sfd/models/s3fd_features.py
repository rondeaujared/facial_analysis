import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_s3fd import s3fd


class s3fd_features(s3fd):
    """
    For a 640x640 pixel image, anchor information:
    Layer  Scale   #     %
    conv3_3   16 25600 75.02    <-- Too many, and too small for practical age estimation
    conv4_3   32 6400  18.76    <-- Probably too small
    conv5_3   64 1600   4.69    <-- Start here!
    conv_fc7 128 400    1.17
    conv6_2  256 100    0.29
    conv7_2  512 25     0.07
    """
    def __init__(self, base_weights, drop_rate, aux_weights=None):
        super(s3fd_features, self).__init__()
        self.weights = base_weights
        self.load_weights()
        self.drop_rate = drop_rate
        self.conv5_3_adult = nn.Conv2d(512, 256//2, kernel_size=3, stride=1, padding=5)
        self.fc7_adult     = nn.Conv2d(1024, 512//2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_adult = nn.Conv2d(512, 256//2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_adult = nn.Conv2d(256, 128//2, kernel_size=3, stride=1, padding=1)
        self.conv_adult    = nn.Conv2d(1152//2, 576//2, kernel_size=3, stride=1, padding=1)
        # self.binary1       = nn.Linear(4712, 512)
        self.binary1 = nn.Linear(5184//2, 2)
        #self.binary2       = nn.Linear(1024, 2)

        if aux_weights:
            print(f"Loading aux_weights {aux_weights}")
            dic = torch.load(aux_weights)
            ndic = {}
            for k, v in dic.items():
                new_key = k.replace('module.', '')
                ndic[new_key] = dic[k]

            self.load_state_dict(ndic)

        for param in self.parameters():
            param.requires_grad = False

        #todo = [self.conv5_3_adult, self.fc7_adult, self.conv6_2_adult, self.conv7_2_adult, self.binary1, self.binary2]
        todo = [self.conv5_3_adult, self.fc7_adult, self.conv6_2_adult, self.conv7_2_adult, self.conv_adult,
                self.binary1]#, self.binary2]
        for layer in todo:
            for param in layer.parameters():
                param.requires_grad = True

    def load_weights(self):
        self.load_state_dict(torch.load(self.weights))

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        adult1 = F.relu(self.conv5_3_adult(f5_3))
        adult1 = F.max_pool2d(adult1, 8, 8)

        adult2 = F.relu(self.fc7_adult(ffc7))
        adult2 = F.max_pool2d(adult2, 4, 4)

        adult3 = F.relu(self.conv6_2_adult(f6_2))
        adult3 = F.max_pool2d(adult3, 2, 2)

        adult4 = F.relu(self.conv7_2_adult(f7_2))
        # adult4 = F.max_pool2d(adult4, 2, 2) dont need
        adult_conv = torch.cat((adult1, adult2, adult3, adult4), dim=1)
        adult_conv = F.relu(self.conv_adult(adult_conv))
        adult_conv = F.max_pool2d(adult_conv, 2, 2)
        binary = adult_conv.view(x.shape[0], -1)
        if self.drop_rate > 0:
            binary = F.dropout(binary, p=self.drop_rate, training=self.training)
        t = self.binary1(binary)
        #t = F.relu(t)
        #if self.drop_rate > 0:
        #    t = F.dropout(t, p=self.drop_rate, training=self.training)
        #t = self.binary2(t)
        padult = F.log_softmax(t, dim=1)


        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        bmax  = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1  = torch.cat([bmax, chunk[3]], dim=1)

        '''
        binary = torch.cat((cls2.view(x.shape[0], -1), cls3.view(x.shape[0], -1),
                            cls4.view(x.shape[0], -1), cls5.view(x.shape[0], -1), cls5.view(x.shape[0], -1)), dim=1)

        if self.drop_rate > 0:
            binary = F.dropout(binary, p=self.drop_rate, training=self.training)
        t = F.relu(self.binary1(binary))
        if self.drop_rate > 0:
            t = F.dropout(t, p=self.drop_rate, training=self.training)
        t = self.binary2(t)
        padult = F.log_softmax(t, dim=1)
        '''

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6, padult]
