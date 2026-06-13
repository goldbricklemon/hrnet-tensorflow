# -*- coding: utf-8 -*-
"""
Created on 27.08.20

@author: ludwikat/einfalmo

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, BatchNormalizationV2, UpSampling2D, \
    ZeroPadding2D, Activation, Add, Concatenate
from hrnet_tf.model.base.blocks import BN_MOMENTUM, BottleneckBlock, BasicBlock


class Branch:
    def __init__(self, block_type, num_blocks, in_channels, channels, stride=1, name="branch", **kwargs):
        self.name = name
        downsample = None

        if stride != 1 or in_channels != channels * block_type.expansion:
            downsample = [Conv2D(channels * block_type.expansion, kernel_size=1, strides=stride, use_bias=False,
                                 name=self.name + f"/{block_type.name_stem}_1/conv2d_sc"),
                          BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5,
                                               name=self.name + f"/{block_type.name_stem}_1/bn_sc")]

        self.branch = []
        self.branch.append(block_type(channels=channels, stride=stride, downsample=downsample,
                                      name=self.name + f"/{block_type.name_stem}_1"))
        for i in range(1, num_blocks):
            self.branch.append(block_type(channels=channels, stride=1, downsample=None,
                                          name=self.name + f"/{block_type.name_stem}_{i + 1}"))

    def __call__(self, inputs):
        x = inputs

        for block in self.branch:
            x = block(x)

        return x


class FuseLayer:
    """
    Exchange features between two layers. This layer is used for exchange between already existing branches and not for new branches
    """

    def __init__(self, in_channels_per_branch, in_branch_idx, out_branch_idx, name="fuse", **kwargs):
        self.name = name
        self.steps = []
        if in_branch_idx < out_branch_idx:  # fuse layer from large to small spatial resolution
            i = -1
            for i in range(out_branch_idx - in_branch_idx - 1):
                self.steps.append(ZeroPadding2D((1, 1), name=self.name + f"/zero_pad_{i + 1}"))
                self.steps.append(
                    Conv2D(in_channels_per_branch[in_branch_idx], kernel_size=3, strides=2, padding='valid',
                           use_bias=False, name=self.name + f"/conv2d_{i + 1}"))
                self.steps.append(
                    BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + f"/bn_{i + 1}"))
                self.steps.append(Activation("relu", name=self.name + f"/relu_{i + 1}"))
            i += 1
            self.steps.append(ZeroPadding2D((1, 1), name=self.name + f"/zero_pad_{i + 1}"))
            self.steps.append(Conv2D(in_channels_per_branch[out_branch_idx], kernel_size=3, strides=2, padding='valid',
                                     use_bias=False, name=self.name + f"/conv2d_{i + 1}"))
            self.steps.append(BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + f"/bn_{i + 1}"))
        elif in_branch_idx > out_branch_idx:  # fuse layer from small to large spatial resolution
            self.steps.append(Conv2D(in_channels_per_branch[out_branch_idx], kernel_size=1, strides=1, padding='valid',
                                     use_bias=False, name=self.name + f"/conv2d_1"))
            self.steps.append(BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + f"/bn_1"))
            self.steps.append(UpSampling2D(size=2 ** (in_branch_idx - out_branch_idx), interpolation='nearest',
                                           name=self.name + f"/upsample_1"))

    def __call__(self, inputs):
        x = inputs

        for step in self.steps:
            x = step(x)

        return x


class HRModule:
    """
    A module consists of a certain number of branches and one fuse layer
    """

    def __init__(self, block_type, num_branches, num_blocks, in_channels_per_branch, channels_per_branch,
                 multi_scale_output=True, concat_output=False, name="hrmodule", **kwargs):
        self.concat_output = concat_output
        self.name = name
        self.branches = []
        self.fuse_layers = []
        for i in range(num_branches):
            self.branches.append(Branch(block_type, num_blocks, in_channels_per_branch[i], channels_per_branch[i],
                                        name=self.name + f"/branch_{i + 1}"))
        if num_branches > 1:
            for out_branch_idx in range(num_branches if multi_scale_output else 1):
                out_branch_fuse_layers = []
                for in_branch_idx in range(num_branches):
                    out_branch_fuse_layers.append(FuseLayer(in_channels_per_branch, in_branch_idx, out_branch_idx,
                                                            name=self.name + f"/fuse_{in_branch_idx}_{out_branch_idx}"))
                self.fuse_layers.append(out_branch_fuse_layers)
        self.in_channels_per_branch = channels_per_branch * block_type.expansion

    def __call__(self, inputs):
        x = inputs
        x_branches = []
        if len(self.branches) == 1:
            return [self.branches[0](x[0])]

        for i in range(len(self.branches)):
            x_branches.append(self.branches[i](x[i]))

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x_branches[0] if i == 0 else self.fuse_layers[i][0](x_branches[0])
            if self.concat_output:
                concat_layers = [y]
            for j in range(1, len(self.branches)):
                if not self.concat_output:
                    # Compute the fuse output for each branch by summation across branches
                    add = Add(name=self.name+f"/add_{i+1}_{j+1}")
                    if i == j:
                        y = add([y, x_branches[j]])
                    else:
                        y = add([y, self.fuse_layers[i][j](x_branches[j])])
                else:
                    # Compute the fuse output for each branch by concatenation across branches
                    if i == j:
                        concat_layers.append(x_branches[j])
                    else:
                        concat_layers.append(self.fuse_layers[i][j](x_branches[j]))
            if self.concat_output:
                y = Concatenate(axis=-1, name=self.name+f"/concat_{i+1}")(concat_layers)

            x_fuse.append(Activation("relu", name=self.name+f"/relu_{i+1}")(y))

        return x_fuse

    def get_in_channels_per_branch(self):
        return self.in_channels_per_branch


class Stem:
    def __init__(self, name="stem", **kwargs):
        self.name = name
        self.pad1 = ZeroPadding2D((1, 1), name=self.name + "/zero_pad_1")
        self.conv1 = Conv2D(64, kernel_size=3, strides=2, padding='valid', use_bias=False, name=self.name + "/conv2d_1")
        self.bn1 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + "/bn_1")
        self.pad2 = ZeroPadding2D((1, 1), name=self.name + "/zero_pad_2")
        self.conv2 = Conv2D(64, kernel_size=3, strides=2, padding='valid', use_bias=False, name=self.name + "/conv2d_2")
        self.bn2 = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + "/bn_2")

    def __call__(self, inputs):
        x = self.pad1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = Activation("relu", name=self.name + "/relu_1")(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = Activation("relu", name=self.name + "/relu_2")(x)

        return x


class Layer1:
    def __init__(self, channels, num_blocks, name="layer1", **kwargs):
        self.name = name
        self.in_channels = 64
        assert self.in_channels != channels * BottleneckBlock.expansion

        downsample = [Conv2D(channels * BottleneckBlock.expansion, kernel_size=1, strides=1, use_bias=False,
                             name=self.name + f"/{BottleneckBlock.name_stem}_1/conv2d_sc"),
                      BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5,
                                           name=self.name + f"/{BottleneckBlock.name_stem}_1/bn_sc")]
        self.blocks = []
        self.blocks.append(
            BottleneckBlock(channels, downsample=downsample, name=self.name + f"/{BottleneckBlock.name_stem}_1"))
        for i in range(1, num_blocks):
            self.blocks.append(BottleneckBlock(channels, name=self.name + f"/{BottleneckBlock.name_stem}_{i + 1}"))

    def __call__(self, inputs):
        x = inputs

        for block in self.blocks:
            x = block(x)

        return x


class TransitionLayer:
    """
    Transition from a single branch to a new branch
    """

    def __init__(self, channels_per_branch_pre_layer, channels_per_branch_cur_layer, branch_idx, name="transition", **kwargs):
        self.name = name
        num_branches_pre = len(channels_per_branch_pre_layer)
        self.steps = []
        if branch_idx < num_branches_pre:
            if channels_per_branch_cur_layer[branch_idx] != channels_per_branch_pre_layer[branch_idx]:
                self.steps.append(ZeroPadding2D((1, 1), name=self.name + "/zero_pad_1"))
                self.steps.append(
                    Conv2D(channels_per_branch_cur_layer[branch_idx], kernel_size=3, strides=1, padding='valid',
                           use_bias=False, name=self.name + "/conv2d_1"))
                self.steps.append(BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + "/bn_1"))
                self.steps.append(Activation("relu", name=self.name + "/relu_1"))
        else:
            for j in range(branch_idx - num_branches_pre + 1):
                channels = channels_per_branch_cur_layer[branch_idx] if j == branch_idx - num_branches_pre else \
                    channels_per_branch_pre_layer[branch_idx]
                self.steps.append(ZeroPadding2D((1, 1), name=self.name + f"/zero_pad_{j + 1}"))
                self.steps.append(Conv2D(channels, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                         name=self.name + f"/conv2d_{j + 1}"))
                self.steps.append(
                    BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + f"/bn_{j + 1}"))
                self.steps.append(Activation("relu", name=self.name + f"/relu_{j + 1}"))

    def __call__(self, inputs):
        x = inputs

        for step in self.steps:
            x = step(x)

        return x


def make_transitions(channels_per_branch_pre_layer, channels_per_branch_cur_layer, name="transition"):
    num_branches_cur = len(channels_per_branch_cur_layer)
    transitions = []
    for branch_idx in range(num_branches_cur):
        transition = TransitionLayer(channels_per_branch_pre_layer, channels_per_branch_cur_layer, branch_idx,
                                     name=name + f"_{branch_idx + 1}")
        if len(transition.steps) > 0:
            transitions.append(transition)
        else:
            transitions.append(None)
    return transitions


class Stage:

    def __init__(self, block_type, num_modules, num_branches, num_blocks, in_channels_per_branch, channels_per_branch,
                 multi_scale_output=True, final_fuse_concat=False, name="stage", **kwargs):
        self.name = name
        self.modules = []
        self.num_branches = num_branches

        for i in range(num_modules):
            curr_multi_scale = False if not multi_scale_output and i == num_modules - 1 else True
            final_concat = final_fuse_concat and i == num_modules - 1
            self.modules.append(HRModule(block_type=block_type, num_branches=num_branches, num_blocks=num_blocks,
                                         in_channels_per_branch=in_channels_per_branch,
                                         channels_per_branch=channels_per_branch,
                                         multi_scale_output=curr_multi_scale,
                                         concat_output=final_concat,
                                         name=self.name + f"/hr_module_{i + 1}"))
            in_channels_per_branch = self.modules[-1].get_in_channels_per_branch()
        self.channels_per_branch = in_channels_per_branch

    def __call__(self, inputs):
        x = inputs

        for module in self.modules:
            x = module(x)

        return x

    def get_channels_per_branch(self):
        return self.channels_per_branch

    def get_num_branches(self):
        return self.num_branches
