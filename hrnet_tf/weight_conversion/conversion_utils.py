# -*- coding: utf-8 -*-
"""
Created on 21 Jan 2021, 13:58

@author: goldbricklemon
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import python as tfpy
import torch
from collections import OrderedDict


def recursive_layers(model_or_layer):
    all_layers = []
    if hasattr(model_or_layer, "layers"):
        inner_layers = model_or_layer.layers
    else:
        inner_layers = model_or_layer._layers
    for inner_layer in inner_layers:
        if not isinstance(inner_layer, tfpy.training.tracking.data_structures.ListWrapper):
            all_layers.append(inner_layer)
        all_layers.extend(recursive_layers(inner_layer))

    return all_layers


def conv_weights_pytorch_to_tf(conv_weights):
    # Convert from (out_c,in_c,h,w) to (h,w,in_c,out_c)
    return np.transpose(np.copy(conv_weights), (2, 3, 1, 0))


def load_pytorch_weights(tf_model,
                         weight_file,
                         branches_per_stage=[1, 2, 3, 4],
                         modules_per_stage=[0, 1, 4, 3],
                         blocks_per_stage=[4, 4, 4, 4],
                         multi_resolution_output=False,
                         include_pose_head=False,
                         blocks_in_pose_head=4,
                         verbose=1):

    num_stages = len(branches_per_stage)

    assert os.path.exists(weight_file), os.path.abspath(weight_file)
    # Load original weights from pytorch model
    weight_dict = torch.load(weight_file, torch.device('cpu'))
    tf_layers = recursive_layers(tf_model)
    tf_layer_names = [layer.name for layer in tf_layers]

    def get_layer(layers, name):
        for layer in layers:
            if layer.name == name:
                return layer
        return None

    # Collect pytorch -> tensorflow layer name mappings
    # Manage separate collections for different layer types
    conv2d_mappings = OrderedDict()
    bn_mappings = OrderedDict()
    conv2d_transpose_mappings = OrderedDict()

    def add_mapping(p_name, t_name):
        mappings = None
        if "conv2d_trans" in t_name:
            mappings = conv2d_transpose_mappings
        elif "conv2d" in t_name:
            mappings = conv2d_mappings
        elif "bn" in t_name:
            mappings = bn_mappings
        assert mappings is not None, "Unexpected tf-layer: " + t_name
        assert p_name not in mappings.keys(), f"pytorch-layer {p_name} already assigned"
        mappings[p_name] = t_name

    # Here.We.Go.
    # Stem: Two strided conv layers
    for k in range(2):
        for a, b in [("conv", "conv2d"), ("bn", "bn")]:
            p = f"{a}{k+1}"
            t = f"stem/{b}_{k+1}"
            add_mapping(p, t)

    # Stage 1: Bottleneck Blocks
    for i in range(blocks_per_stage[0]):
        # Each bottleneck has three conv layers, first one has additional layers on shortcut
        for k in range(3):
            for a, b in [("conv", "conv2d"), ("bn", "bn")]:
                p = f"layer1.{i}.{a}{k+1}"
                t = f"stage_1/bottleneck_{i+1}/{b}_{k+1}"
                add_mapping(p, t)
        if i == 0:
            for a, b in [("downsample.0", "conv2d_sc"), ("downsample.1", "bn_sc")]:
                p = f"layer1.{i}.{a}"
                t = f"stage_1/bottleneck_{i+1}/{b}"
                add_mapping(p, t)

    # Transition 1: Transition from one to two branches
    num_t = 1
    for i_out in range(2):
        for a, b in [("0", "conv2d_1"), ("1", "bn_1")]:
            if i_out > 0:
                a = "0." + a
            p = f"transition{num_t}.{i_out}.{a}"
            t = f"transition_{num_t}_{i_out + 1}/{b}"
            add_mapping(p, t)

    def add_stage_mapping(stage, modules, branches, blocks, full_final_fusion=True):
        s = stage
        # Stage s: HRModule with branches
        for i in range(modules):
            # Branches
            for j in range(branches):
                # Each branch has four basic blocks
                # We assume basic blocks here !!!
                for k in range(blocks):
                    # Each basic block has two conv/bn layer stacks
                    for l in range(2):
                        for a, b in [("conv", "conv2d"), ("bn", "bn")]:
                            p = f"stage{s}.{i}.branches.{j}.{k}.{a}{l+1}"
                            t = f"stage_{s}/hr_module_{i+1}/branch_{j+1}/basic_block_{k+1}/{b}_{l+1}"
                            add_mapping(p, t)

            # Fusion with branches
            if not full_final_fusion and i == modules - 1:
                out_branches = [0]
            else:
                out_branches = list(range(branches))
            for j_out in out_branches:
                for j_in in range(branches):
                    if j_out == j_in:
                        pass
                    elif j_out < j_in:
                        for a, b in [("0", "conv2d_1"), ("1", "bn_1")]:
                            p = f"stage{s}.{i}.fuse_layers.{j_out}.{j_in}.{a}"
                            t = f"stage_{s}/hr_module_{i+1}/fuse_{j_in}_{j_out}/{b}"
                            add_mapping(p, t)
                    else: #j_out > j_in
                        for k in range(j_out - j_in):
                            for a, b in [("0", "conv2d"), ("1", "bn")]:
                                p = f"stage{s}.{i}.fuse_layers.{j_out}.{j_in}.{k}.{a}"
                                t = f"stage_{s}/hr_module_{i + 1}/fuse_{j_in}_{j_out}/{b}_{k+1}"
                                add_mapping(p, t)

    def add_transition_mapping(transition):
        num_t = transition
        # Transition t: Transition from t to t+1 branches:
        for a, b in [("0", "conv2d_1"), ("1", "bn_1")]:
            # Slightly hard-coded here, but transition 2 and above only consists of one conv and bn
            p = f"transition{num_t}.{num_t}.0.{a}"
            t = f"transition_{num_t}_{num_t + 1}/{b}"
            add_mapping(p, t)

    for stage, (branches, modules, blocks) in enumerate(zip(branches_per_stage[1:], modules_per_stage[1:], blocks_per_stage[1:])):
        # Compensate for 1-based index and slicing
        s = stage + 2
        # Stage s
        full_fusion = multi_resolution_output or s < num_stages
        add_stage_mapping(stage=s, modules=modules, branches=branches, blocks=blocks, full_final_fusion=full_fusion)
        # Transition s->s+1
        if s < num_stages:
            add_transition_mapping(transition=s)

    # Handle HigherHRNet pose head
    if include_pose_head:
        # Pose head: conv, concat, transpose conv (+bn+relu), basic blocks, conv
        # Handle first, last conv and transpose conv
        for a, b in [("final_layers.0", "conv2d_1"),
                     ("deconv_layers.0.0.0", "conv2d_trans_1"),
                     ("deconv_layers.0.0.1", "bn_1"),
                     ("final_layers.1", "conv2d_2")]:
            p = a
            t = f"pose_head/{b}"
            add_mapping(p, t)

        # Handle basic blocks
        for k in range(blocks_in_pose_head):
            # Each basic block has two conv/bn layer stacks
            for l in range(2):
                for a, b in [("conv", "conv2d"), ("bn", "bn")]:
                    p = f"deconv_layers.0.{k + 1}.0.{a}{l + 1}"
                    t = f"pose_head/basic_block_{k + 1}/{b}_{l + 1}"
                    add_mapping(p, t)


    # Track which weights were used from the weight file
    weights_used = {name: False for name in weight_dict.keys()}
    # Track which tf layers have assigned weights
    layers_assigned = {name: False for layer, name in zip(tf_layers, tf_layer_names) if
                       type(layer) in [tf.keras.layers.Conv2D,
                                 tf.keras.layers.Conv2DTranspose,
                                 tf.keras.layers.BatchNormalization,
                                 tfpy.keras.layers.BatchNormalizationV2,
                                 tfpy.keras.layers.BatchNormalizationV1]
                       }

    # Assign conv2D and conv2d_transpose weights
    for p_name, t_name in list(conv2d_mappings.items()) + list(conv2d_transpose_mappings.items()):
        t_layer = get_layer(tf_layers, t_name)
        assert t_layer is not None, f"Can't find tf layer: {t_name}"
        assert layers_assigned[t_name] is False
        t_weight_list = []
        # The order is important here. It has to match the weight order of t_layer.get_weights()
        for suffix in [".weight", ".bias"]:
            p_weight_name = p_name + suffix
            if p_weight_name in weight_dict.keys():
                assert weights_used[p_weight_name] is False
                p_weights = weight_dict[p_weight_name].cpu().numpy()
                is_bias = suffix == ".bias"
                if is_bias:
                    t_weights = p_weights
                else:
                    t_weights = conv_weights_pytorch_to_tf(p_weights)

                t_weight_list.append(t_weights)
                weights_used[p_weight_name] = True

        # Check for compatible shapes
        cur_weights = t_layer.get_weights()
        assert len(cur_weights) == len(t_weight_list)
        for cw, tw in zip(cur_weights, t_weight_list):
            assert cw.shape == tw.shape

        # Apply all weights for this layer
        t_layer.set_weights(t_weight_list)
        layers_assigned[t_name] = True

    # Assign batchnorm statistics and weights
    for p_name, t_name in bn_mappings.items():
        t_layer = get_layer(tf_layers, t_name)
        assert t_layer is not None, f"Can't find tf layer: {t_name}"
        assert layers_assigned[t_name] is False
        t_weight_list = []
        # The order is important here. It has to match the weight order of t_layer.get_weights()
        for weight_name in ["weight", "bias", "running_mean", "running_var"]:
            key = p_name + "." + weight_name
            assert weights_used[key] is False
            p_weight = weight_dict[key].cpu().numpy()
            weights_used[key] = True
            t_weight_list.append(p_weight)

        # Check for compatible shape
        cur_weights = t_layer.get_weights()
        assert len(cur_weights) == 4
        for cw, tw in zip(cur_weights, t_weight_list):
            assert cw.shape == tw.shape

        # Apply all weights for this layer
        t_layer.set_weights(t_weight_list)
        layers_assigned[t_name] = True

    # Print unused weights and non-assigned layers
    unused = [key for key, used in weights_used.items() if used is False and "num_batches_tracked" not in key]
    unassigned = [layer_name for layer_name, assigned in layers_assigned.items() if assigned is False]
    print(f"{len(unused)} weights from source weightfile are unused")
    if verbose:
        for weight_name in unused:
            print(weight_name)

    print(f"{len(unassigned)} layers in target tensorflow model were not assigned any weights")
    if verbose:
        for layer_name in unassigned:
            print(layer_name)
