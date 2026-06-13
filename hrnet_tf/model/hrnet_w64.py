# -*- coding: utf-8 -*-
"""
Created on 03 Feb 2021, 10:12

@author: goldbricklemon
"""

from hrnet_tf.model.hrnet import hrnet, HRNetConfig


class HRNetW64Config(HRNetConfig):
    """
    Configuration for HRNet-w64
    """
    HRNET_NAME = "HRNet-w64"
    # List of list; for each stage, the #channels per branch
    CHANNELS_PER_STAGE = [[64],
                          [64, 128],
                          [64, 128, 256],
                          [64, 128, 256, 512],
                          ]
    # List; for each stage, the #HR-modules (same across all branches)
    MODULES_PER_STAGE = [0, 1, 4, 3]
    # Number of blocks in each stage (the same for each module in a stage)
    BLOCKS_PER_STAGE = [4, 4, 4, 4]
    # Basic building block in all HR-modules
    # Has to be "BasicBlock" or "BottleneckBlock" (untested!)
    BLOCK_TYPE = "BasicBlock"
    # Whether to compute (and return) the output of all branches
    # after the final resolution fusion, or just the one with highest resolution (i.e. the first branch)
    # Note: This changes the architecture of the final fusion layer
    MULTI_RESOLUTION_OUTPUT = True


def hrnet_w64(input_tensor=None, input_shape=None, weights=None, **kwargs):
    config = HRNetW64Config()
    if "name" in kwargs.keys():
        model_name = kwargs["name"]
    else:
        model_name = config.HRNET_NAME
    model = hrnet(config=config,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  weights=weights,
                  model_name=model_name,
                  **kwargs)
    return model
