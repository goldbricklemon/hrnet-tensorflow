# -*- coding: utf-8 -*-
"""
Created on 03.02.2021

@author: einfalmo

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, BatchNormalizationV2, Conv2DTranspose, concatenate, Activation
from hrnet_tf.model.base.blocks import BN_MOMENTUM, BasicBlock
from hrnet_tf.model.hrnet_w32 import HRNetW32Config
from hrnet_tf.model.hrnet import hrnet


class PoseHigherHRNetConfig(HRNetW32Config):
    # Exemplary configuration for PoseHigherHRNet
    # So far, only HRNet-w32 backend is supported!
    MULTI_RESOLUTION_OUTPUT = False
    # Number of joints, here from COCO
    NUM_JOINTS = 17
    # One tag map per joint?
    TAG_PER_JOINT = True
    # Generate 1/2 resolution output
    HIGH_RES_OUTPUT = True
    # Architecture has two heatmap output layers (at 1/4 and 1/2 of input resolution)
    # Specify, which output layers should additionally generate tag maps
    USE_ASSOCIATIVE_EMBEDDING_LOSS = [True, False]
    # Number of deconv channels after concatenating backbone output and 1/4-resolution heatmaps
    DECONV_CHANNELS = 32
    # Number of blocks after upsampling in pose head
    BLOCKS_IN_HEAD = 4


def pose_higherhrnet(config,
                     input_tensor=None,
                     input_shape=None,
                     model_name=None):
    assert not (input_tensor is None and input_shape is None), \
        "You have to specify either an input shape or an input tensor"
    if input_tensor is None:
        assert len(input_shape) == 3
        input = keras.layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    hrnet_backbone = hrnet(config=config,
                           input_tensor=input,
                           model_name=config.HRNET_NAME.lower())

    pose_head = FinalModule(num_joints=config.NUM_JOINTS,
                            tag_per_joint=config.TAG_PER_JOINT,
                            include_ae_output=config.USE_ASSOCIATIVE_EMBEDDING_LOSS,
                            high_res_output=config.HIGH_RES_OUTPUT,
                            deconv_channels=config.DECONV_CHANNELS,
                            num_blocks=config.BLOCKS_IN_HEAD,
                            name="pose_head")

    x = hrnet_backbone.output
    outputs = pose_head(x)
    model = keras.Model(inputs=[input], outputs=outputs, name=model_name)
    return model


class FinalModule:

    def __init__(self, num_joints, tag_per_joint, include_ae_output, high_res_output, deconv_channels, num_blocks,
                 name="pose_head", **kwargs):
        self.name = name
        self.generate_high_res_output = high_res_output
        dim_tag = num_joints if tag_per_joint else 1

        output_channels = num_joints + dim_tag if include_ae_output[0] else num_joints
        self.output_conv1 = Conv2D(output_channels, kernel_size=1, strides=1, padding="valid",
                                   name=self.name + "/conv2d_1")

        if self.generate_high_res_output:
            self.deconv = Conv2DTranspose(deconv_channels, kernel_size=4, strides=2, padding="same",
                                          output_padding=None,
                                          use_bias=False, name=self.name + "/conv2d_trans_1")
            self.bn = BatchNormalizationV2(momentum=BN_MOMENTUM, epsilon=1e-5, name=self.name + "/bn_1")
            self.relu = Activation("relu", name=self.name + "/relu_1")

            self.blocks = []
            for i in range(num_blocks):
                self.blocks.append(BasicBlock(deconv_channels, name=self.name + f"/basic_block_{i + 1}"))

            output_channels = num_joints + dim_tag if include_ae_output[1] else num_joints
            self.output_conv2 = Conv2D(output_channels, kernel_size=1, strides=1, padding="valid",
                                       name=self.name + "/conv2d_2")

    def __call__(self, inputs):
        x = inputs

        final_outputs = []

        y = self.output_conv1(x)
        final_outputs.append(y)

        if self.generate_high_res_output:
            x = concatenate([x, y])

            x = self.deconv(x)
            x = self.bn(x)
            x = self.relu(x)

            for block in self.blocks:
                x = block(x)

            y = self.output_conv2(x)
            final_outputs.append(y)

        return final_outputs


def select_layer_regex(layer_selector):
    layer_regex_options = {
        # all layers but the backbone
        "pose_head": r"(pose_head.*)",
        # From a specific stage and up
        "1+":        r"(stage_1.*)|(transition_1.*)|(stage_2.*)|(transition_2.*)|(stage_3.*)|(transition_3.*)|(stage_3.*)|(stage_4.*)|(pose_head.*)",
        "2+":        r"(stage_2.*)|(transition_2.*)|(stage_3.*)|(transition_3.*)|(stage_4.*)|(pose_head.*)",
        "3+":        r"(stage_3.*)|(transition_3.*)|(stage_4.*)|(pose_head.*)",
        "4+":        r"(stage_4.*)|(pose_head.*)",
        # All layers
        "all":       ".*",
    }
    if layer_selector in layer_regex_options.keys():
        layer_regex = layer_regex_options[layer_selector]
    else:
        raise Exception(f"Invalid layer selector: {layer_selector}")
    return layer_regex



# if __name__ == '__main__':
#     import os
#     import numpy as np
#     import cv2
#     import matplotlib.pyplot as plt

#     input_size = (512, 512, 3)
#     config = PoseHigherHRNetConfig()
#     weights = "converted_models/pose_higher_hrnet/pose_higher_hrnet_w32_512_coco.h5"
#     model = pose_higherhrnet(config=config, input_shape=input_size, model_name="pose_higherhrnet")
#     model.load_weights(weights, by_name=True)

#     image_file = "SOME_IMAGE.jpg"
#     mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
#     std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
#     num_joints = 17

#     # Normalize to [0,1]
#     orig_image = cv2.imread(image_file)[:, :, ::-1].astype(np.float32) / 255.
#     resized = orig_image[np.newaxis, ...]
#     resized = tf.image.resize_with_pad(resized, input_size[0], input_size[1])
#     resized_np = resized.numpy()[0]
#     # Normalize by mean and std
#     norm_image = (resized - tf.reshape(mean, shape=(1, 1, -1))) / tf.reshape(std, shape=(1, 1, -1))

#     out_low_res, out_high_res = model(norm_image, training=False)

#     plt.figure(figsize=(5, 12))
#     plt.subplot(3, 1, 1)
#     plt.imshow(resized_np, vmin=0, vmax=1)

#     plt.subplot(3, 1, 2)
#     # Aggregate tagmaps
#     low_res_heatmap = np.max(out_low_res.numpy()[0, :, :, num_joints:], axis=2)
#     plt.imshow(low_res_heatmap)  # , vmin=0, vmax=1)

#     plt.subplot(3, 1, 3)
#     # Aggregate low-res heatmaps
#     high_res_heatmap = np.max(out_low_res.numpy()[0, :, :, :num_joints], axis=2)
#     plt.imshow(high_res_heatmap, vmin=0, vmax=1)

#     plt.show()

#     print(model.name)
