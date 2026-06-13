# -*- coding: utf-8 -*-
"""
Created on 03 Feb 2021, 10:12

@author: goldbricklemon
"""

from tensorflow import keras
from hrnet_tf.model.base.blocks import BasicBlock, BottleneckBlock
from hrnet_tf.model.base.modules import Stem, Layer1, Stage, make_transitions


def base_hrnet(channels_per_stage,
               modules_per_stage,
               blocks_per_stage,
               num_stages,
               block_type,
               multi_resolution_output,
               final_fuse_concat=False,
               input_tensor=None,
               input_shape=None,
               model_name=None,
               **kwargs):
    """
    Base HRNet architecture.
    :param channels_per_stage: list of list; for each stage, the #channels per branch
    :param modules_per_stage: list; for each stage, the #HR-modules (same across all branches)
    :param blocks_per_stage: list, #blocks in each stage (i.e. all modules in one stage have the same)
    :param num_stages: int, #blocks to create
    :param block_type: basic building block in all HR-modules
    :param multi_resolution_output: Whether to compute (and return) the output of all branches
    after the final resolution fusion, or just the one with highest resolution (i.e. the first branch)
    :param final_fuse_concat: Default: False. Final fuse layer adds the feature maps of all branches after channel mapping.
    Set to True, to concatenate the features instead of adding them.
    Note: This changes the architecture of the final fusion layer.
    With True, the final output of each branch has 4x the number of channels.
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model.
    :param input_shape: optional shape tuple. It should have exactly 3 inputs channels.
    """
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

    # Create building blocks first
    assert len(channels_per_stage) == len(modules_per_stage) == len(blocks_per_stage)
    # Only create as many blocks as requested
    num_stages = min(num_stages, len(channels_per_stage))

    if type(block_type) is str:
        if block_type == "BasicBlock":
            block_type = BasicBlock
        elif block_type == "BottleneckBlock":
            block_type = BottleneckBlock
        else:
            raise Exception(f"Block type [{block_type}] is not supported")
    else:
        assert any([isinstance(block_type, B) for B in [BasicBlock, BottleneckBlock]])

    # Stage 1 is special
    stage1_in_channel = channels_per_stage[0][0]
    stage1_out_channel = channels_per_stage[0][0] * 4
    stem = Stem(name="stem")
    stage1 = Layer1(stage1_in_channel, blocks_per_stage[0], name="stage_1")

    transitions = []
    stages = []
    # Create transition -> stage -> transition -> stage -> .... -> transition -> stage
    for stage in range(1, num_stages):
        is_final_stage = stage == num_stages - 1
        stage_channels = channels_per_stage[stage]
        stage_modules = modules_per_stage[stage]
        in_stage_channels = [c * block_type.expansion for c in stage_channels]
        # First transition is special, has to map stage1 channels differently
        if stage == 1:
            transition_in = [stage1_out_channel]
        else:
            transition_in = stages[-1].get_channels_per_branch()

        transition = make_transitions(transition_in, in_stage_channels, name=f"transition_{stage}")
        multi_res = (not is_final_stage) or multi_resolution_output
        final_concat = is_final_stage and final_fuse_concat
        stage = Stage(block_type=block_type, num_modules=stage_modules, num_branches=len(stage_channels),
                      num_blocks=blocks_per_stage[stage],
                      in_channels_per_branch=in_stage_channels, channels_per_branch=stage_channels,
                      multi_scale_output=multi_res,
                      final_fuse_concat=final_concat,
                      name=f"stage_{stage + 1}")
        transitions.append(transition)
        stages.append(stage)

    # Create functional model from building blocks
    x = stem(input)
    x = stage1(x)

    y_list = [x]

    for transition, stage in zip(transitions, stages):
        x_list = []
        # Apply transition layer for each branch (if needed)
        for i in range(stage.get_num_branches()):
            if transition[i] is not None:
                x_list.append(transition[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = stage(x_list)

    model = keras.Model(inputs=[input], outputs=y_list, name=model_name)
    return model
