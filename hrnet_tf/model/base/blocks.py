# -*- coding: utf-8 -*-
"""
Created on 27.08.20

@author: goldbricklemon

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, BatchNormalizationV2, ZeroPadding2D, Activation, Add


# BatchNorm momentum from the original PyTorch model
BN_MOMENTUM = 0.9


class BasicBlock:
    expansion = 1
    name_stem = "basic_block"

    def __init__(self, channels, stride=1, downsample=None, name=None, **kwargs):
        self.name = name if name is not None else self.name_stem
        self.pad1 = ZeroPadding2D((1, 1), name=self.name+"/zero_pad_1")
        self.conv1 = Conv2D(channels, use_bias=False, kernel_size=3, activation=None, padding='valid', strides=stride, name=self.name+"/conv2d_1")
        self.bn1 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name+"/bn_1")
        self.pad2 = ZeroPadding2D((1, 1), name=self.name + "/zero_pad_2")
        self.conv2 = Conv2D(channels, use_bias=False, kernel_size=3, activation=None, padding='valid', strides=stride, name=self.name+"/conv2d_2")
        self.bn2 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name+"/bn_2")
        self.downsample = downsample

    def __call__(self, inputs):
        shortcut = inputs

        x = self.pad1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        act = Activation("relu", name=self.name+"/relu_1")
        x = act(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            for layer in self.downsample:
                shortcut = layer(shortcut)

        x = Add(name=self.name+"/add")([x, shortcut])
        x = Activation("relu", name=self.name+"/relu_2")(x)

        return x


class BottleneckBlock:
    expansion = 4
    name_stem = "bottleneck"

    def __init__(self, channels, stride=1, downsample=None, name=None, **kwargs):
        self.name = name if name is not None else self.name_stem
        self.conv1 = Conv2D(channels, use_bias=False, kernel_size=1, activation=None, padding='valid', name=self.name+"/conv2d_1")
        self.bn1 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name+"/bn_1")
        self.pad = ZeroPadding2D((1, 1), name=self.name+"/zero_pad_1")
        self.conv2 = Conv2D(channels, use_bias=False, kernel_size=3, activation=None, padding='valid', strides=stride, name=self.name+"/conv2d_2")
        self.bn2 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name+"/bn_2")
        self.conv3 = Conv2D(channels * self.expansion, use_bias=False, kernel_size=1, activation=None, padding='valid', name=self.name+"/conv2d_3")
        self.bn3 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name+"/bn_3")
        self.downsample = downsample

    def __call__(self, inputs):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = Activation("relu", name=self.name+"/relu_1")(x)

        x = self.pad(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = Activation("relu", name=self.name+"/relu_2")(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            for layer in self.downsample:
                shortcut = layer(shortcut)

        x = Add(name=self.name+"/add")([x, shortcut])
        x = Activation("relu", name=self.name+"/relu_3")(x)

        return x


if __name__ == '__main__':
    input = keras.Input(shape=(64, 64, 3))
    output = BasicBlock(channels=3, name="basic_block")(input)

    model = keras.Model(inputs=[input], outputs=[output])
    model.summary()

    in_between = model.get_layer("basic_block/relu_1").output
    model = keras.Model(inputs=[input], outputs=[in_between, output])
    model.summary()
