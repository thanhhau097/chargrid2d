# TODO: Code model in figure 2, init weights

import torch
import torch.nn as nn

from utils import init_weights


class Encoder(nn.Module):
    def __init__(self, input_channels, base_channels=64):
        super(Encoder, self).__init__()
        self.C = base_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, self.C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.C, self.C * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.C * 2, self.C * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.C * 4, self.C * 8, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(self.C * 8, self.C * 8, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        init_weights(self.block1)
        init_weights(self.block2)
        init_weights(self.block3)
        init_weights(self.block4)
        init_weights(self.block5)

    def forward(self, x):
        outs = []

        x = self.block1(x)
        outs.append(x)
        # print('after block 1', x.size())

        x = self.block2(x)
        outs.append(x)
        # print('after block 2', x.size())

        x = self.block3(x)
        outs.append(x)
        # print('after block 3', x.size())

        x = self.block4(x)
        # print('after block 4', x.size())
        x = self.block5(x)
        outs.append(x)
        # print('after block 5', x.size())
        return outs


class SemanticSegmentationDecoder(nn.Module):
    def __init__(self, output_channels, base_channels=64):
        super(SemanticSegmentationDecoder, self).__init__()
        self.output_channels = output_channels
        self.C = base_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(self.C * 12, self.C * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C * 4, self.C * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 4),  # TODO: do we need?
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.C * 6, self.C * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C * 2, self.C * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 2),  # TODO: do we need?
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.C * 3, self.C, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C, self.C, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(self.C),
            # nn.Softmax(dim=1),
        )

        init_weights(self.block1)
        init_weights(self.block2)
        init_weights(self.block3)
        init_weights(self.block4)

    def forward(self, x):
        """

        :param x: A list of 4 outputs from encoder
        :return:
        """
        y = torch.cat([x[2], x[3]], dim=1)
        y = self.block1(y)

        y = torch.cat([x[1], y], dim=1)
        y = self.block2(y)

        y = torch.cat([x[0], y], dim=1)
        y = self.block3(y)
        # print(y.size())

        y = self.block4(y)
        return y  #.permute(0, 2, 3, 1)  # non-softmax


class BoudingBoxRegressionDecoder(nn.Module):
    def __init__(self, n_a=1, base_channels=64):
        """

        :param n_a:﻿ number of anchors per pixel.
        :param base_channels:
        """
        super(BoudingBoxRegressionDecoder, self).__init__()
        self.n_a = n_a
        self.C = base_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(self.C * 12, self.C * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C * 4, self.C * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 4),  # TODO: do we need?
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.C * 6, self.C * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C * 2, self.C * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 2),  # TODO: do we need?
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.C * 3, self.C, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.C, self.C, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

        self.block4e = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, 2 * self.n_a, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(self.C),
            # nn.Softmax(dim=1),
        )

        self.block4f = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, 4 * self.n_a, kernel_size=3, stride=1, padding=1)
        )

        init_weights(self.block1)
        init_weights(self.block2)
        init_weights(self.block3)
        init_weights(self.block4e)
        init_weights(self.block4f)

    def forward(self, x):
        """

        :param x: A list of 4 outputs from encoder
        :return:
        """
        y = torch.cat([x[2], x[3]], dim=1)
        y = self.block1(y)

        y = torch.cat([x[1], y], dim=1)
        y = self.block2(y)

        y = torch.cat([x[0], y], dim=1)
        y = self.block3(y)
        # print(y.size())

        ye = self.block4e(y)
        yf = self.block4f(y)

        return ye, yf  # ye.permute(0, 2, 3, 1), yf.permute(0, 2, 3, 1)  # non-softmax


class Chargrid2D(nn.Module):
    def __init__(self, input_channels, n_classes, n_a=1, base_channels=64):
        super(Chargrid2D, self).__init__()

        self.encoder = Encoder(input_channels, base_channels)
        self.semantic_segmentation_decoder = SemanticSegmentationDecoder(n_classes, base_channels)
        self.bounding_box_regression_decoder = BoudingBoxRegressionDecoder(n_a, base_channels)

    def forward(self, x):
        x = self.encoder(x)
        y1 = self.semantic_segmentation_decoder(x)
        y2, y3 = self.bounding_box_regression_decoder(x)

        return y1, y2, y3


if __name__ == '__main__':
    model = Chargrid2D(input_channels=302, n_classes=10)
    x = torch.ones((1, 302, 512, 512))
    y1, y2, y3 = model(x)
    print(y1.size())
    print(y2.size())
    print(y3.size())

