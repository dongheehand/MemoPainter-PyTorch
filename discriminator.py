## Most parts related to Discriminator are borrowed from https://github.com/awesome-davian/Text2Colors ##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, img_dim, feature_dim, imsize, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()


        input_dim = img_dim + feature_dim

        layers = []
        layers.append(nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(imsize / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(k_size * k_size * curr_dim),
            nn.Linear(k_size * k_size * curr_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, ab_img, l_img, color_feat):
        x = torch.cat([ab_img, l_img, color_feat], dim = 1)
        batch_size = x.size(0)
        h = self.main(x)
        out = self.conv1(h)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out

