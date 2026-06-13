# -*- coding: utf-8 -*-
"""
Created on 03 Feb 2021, 10:12

@author: goldbricklemon
"""

from hrnet_tf.model.base.base_hrnet import base_hrnet


class HRNetConfig:
    """
    Config collection to describe a certain HRNet variant.
    The configuration below is the exemplary HRNet-w32 configuration for image classification
    """
    HRNET_NAME = "HRNet"
    # List of list; for each stage, the #channels per branch
    CHANNELS_PER_STAGE = [[64],
                          [32, 64],
                          [32, 64, 128],
                          [32, 64, 128, 256],
                          ]
    # List; for each stage, the #HR-modules (same across all branches)
    MODULES_PER_STAGE = [0, 1, 4, 3]
    # Number of blocks in each stage (the same for each module in a stage)
    BLOCKS_PER_STAGE = [4, 4, 4, 4]
    # Number of stages to generate. Default is all four stages
    NUM_STAGES = 4
    # Basic building block in all HR-modules
    # Has to be "BasicBlock" or "BottleneckBlock" (untested!)
    BLOCK_TYPE = "BasicBlock"
    # Whether to compute (and return) the output of all branches
    # after the final resolution fusion, or just the one with highest resolution (i.e. the first branch)
    # Note: This changes the architecture of the final fusion layer
    MULTI_RESOLUTION_OUTPUT = True
    # Default (False): Final fuse layer adds the feature maps of all branches after channel mapping.
    # Set to True, to concatenate the features instead of adding them.
    # Note: This changes the architecture of the final fusion layer.
    # With True, the final output of each branch has 4x the number of channels.
    FINAL_FUSE_CONCAT = False


def hrnet(config, input_tensor=None, input_shape=None, model_name=None, weights=None, **kwargs):
    name = model_name if model_name is not None else config.HRNET_NAME
    model = base_hrnet(channels_per_stage=config.CHANNELS_PER_STAGE,
                       modules_per_stage=config.MODULES_PER_STAGE,
                       blocks_per_stage=config.BLOCKS_PER_STAGE,
                       num_stages=config.NUM_STAGES,
                       block_type=config.BLOCK_TYPE,
                       multi_resolution_output=config.MULTI_RESOLUTION_OUTPUT,
                       final_fuse_concat=config.FINAL_FUSE_CONCAT,
                       input_tensor=input_tensor,
                       input_shape=input_shape,
                       model_name=name,
                       **kwargs)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model


if __name__ == '__main__':
    config = HRNetConfig()
    input_shape = (512, 512, 3)
    model = hrnet(config=config, input_shape=input_shape)
    model.summary(line_length=240)
