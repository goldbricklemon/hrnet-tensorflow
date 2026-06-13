# -*- coding: utf-8 -*-
"""
Created on 08 Feb 2021, 15:47

@author: goldbricklemon
"""

import os
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from hrnet_tf.model.hrnet import hrnet
from hrnet_tf.model.hrnet_w32 import hrnet_w32, HRNetW32Config


N_CLASSES = 10


@pytest.fixture
def weights():
    """
    Fixture to load the converted weights file for HRNet-w32
    """
    weights_path = "converted_models/hrnetv2_w32_imagenet_pretrained_notop.h5"
    assert os.path.exists(weights_path), f"Converted weights file not found at {weights_path}"
    return weights_path


def dummy_hrnetw32_classifier(input_tensor=None, input_shape=None, weights=None, **kwargs):

    backbone = hrnet_w32(input_tensor=input_tensor, input_shape=input_shape, weights=weights)
    pooling = keras.layers.GlobalAveragePooling2D()
    classifier = keras.layers.Dense(units=N_CLASSES)

    x = backbone.outputs
    assert len(x) == 4
    # Backbone output consists of 4 branches.
    # We just combine them somehow in a nonsensical manner
    x = [pooling(branch) for branch in x]
    x = keras.layers.Concatenate(axis=-1)(x)
    x = classifier(x)
    model = keras.Model(inputs=backbone.inputs, outputs=[x])
    return model


def test_hrnetw32_classifier_input_tensor():
    """
    Build custom net with HRNet32 backbone, with input_tensor but without weights
    """
    input_shape = (224, 224, 3)
    input_tensor = keras.Input(shape=input_shape)
    model = dummy_hrnetw32_classifier(input_tensor=input_tensor)
    input_data = np.random.normal(loc=0.1, scale=1, size=(2,) + input_shape).astype(np.float32)
    output = model(input_data)
    assert output.shape == (2, N_CLASSES)


def test_hrnetw32_classifier_input_shape():
    """
    Build custom net with HRNet32 backbone, with input_shape but without weights
    """
    input_shape = (224, 224, 3)
    model = dummy_hrnetw32_classifier(input_shape=input_shape)
    input_data = np.random.normal(loc=0.1, scale=1, size=(2,) + input_shape).astype(np.float32)
    output = model(input_data)
    assert output.shape == (2, N_CLASSES)
    model.summary()


def test_hrnetw32_classifier_with_weights(weights):
    """
    Build custom net with HRNet32 backbone, with input_shape and weights
    """
    input_shape = (224, 224, 3)
    model = dummy_hrnetw32_classifier(input_shape=input_shape, weights=weights)
    input_data = np.random.normal(loc=0.1, scale=1, size=(2,) + input_shape).astype(np.float32)
    output = model(input_data)
    assert output.shape == (2, N_CLASSES)
    model.summary()


def test_hrnetw32_classifier_graph_mode(weights):
    """
    Build custom net with HRNet32 backbone, with input_shape and weights
    Run in graph mode
    """
    input_shape = (224, 224, 3)
    model = dummy_hrnetw32_classifier(input_shape=input_shape, weights=weights)
    input_data = np.random.normal(loc=0.1, scale=1, size=(2,) + input_shape).astype(np.float32)
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    @tf.function
    def run_model(input_tensor):
        output = model(input_tensor)
        return output

    # Run multiple times, make sure tracing is run max. two times
    for _ in range(4):
        run_model(input_tensor)
    assert True # Asserting successful execution of the loop


def test_hrnetw32_classifier_multiple_outputs(weights):
    """
    Build custom net with HRNet32 backbone, with input node and multiple in-between outputs
    """
    input_shape = (224, 224, 3)
    input_tensor = keras.Input(shape=input_shape)
    base_model = dummy_hrnetw32_classifier(input_tensor=input_tensor, weights=weights)

    in_between = base_model.get_layer("stage_1/bottleneck_4/relu_3").output

    custom_model = keras.Model(base_model.inputs, outputs=[in_between, base_model.output])
    custom_model.summary()
    input_data = np.random.normal(loc=0.1, scale=1, size=(2,) + input_shape).astype(np.float32)
    outputs = custom_model(input_data)
    assert len(outputs) == 2
    assert outputs[1].shape == (2, N_CLASSES)


def test_hrnetw32_concat_fuse(weights):
    """
    Build HRNet32 backbone, where the output of the last fuse layer uses concatenation instead of summation
    """

    class Test6Config(HRNetW32Config):
        FINAL_FUSE_CONCAT = True
        MULTI_RESOLUTION_OUTPUT = False

    config = Test6Config()
    input_shape = (224, 224, 3)
    model = hrnet(config=config, input_shape=input_shape, weights=weights)
    model.summary()
    assert len(model.outputs) == 1
    assert model.output.shape[3] == config.CHANNELS_PER_STAGE[-1][0] * 4
