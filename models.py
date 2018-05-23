from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class JC3D(nn.Module):
    def __init__(self, input_size=(1, 3, 16, 112, 112), num_classes=101):
        super(JC3D, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=(2, 2, 1)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

        )

        self.flat_fts = self.get_flat_fts(input_size, self.features)

        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftr, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),

            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),

            nn.Linear(2048, num_classes),
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts) # 8192
        out = self.classifier(flat_fts)

        return out


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes = 101):
        super(C3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096), # fc6
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # fc7
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(-1, 8192)
        x = self.classifier(x)

        return x