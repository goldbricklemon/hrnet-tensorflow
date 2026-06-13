# -*- coding: utf-8 -*-
"""
Created on 03 Feb 2021, 14:17

@author: goldbricklemon
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from hrnet_tf.weight_conversion.conversion_utils import load_pytorch_weights
from hrnet_tf.model.hrnet import hrnet
from hrnet_tf.model.hrnet_w32 import hrnet_w32, HRNetW32Config
from hrnet_tf.model.hrnet_w18_small_v1 import HRNetW18SmallV1Config
from hrnet_tf.model.hrnet_w18_small_v2 import HRNetW18SmallV2Config
from hrnet_tf.model.hrnet_w18 import HRNetW18Config
from hrnet_tf.model.hrnet_w30 import HRNetW30Config
from hrnet_tf.model.hrnet_w40 import HRNetW40Config
from hrnet_tf.model.hrnet_w44 import HRNetW44Config
from hrnet_tf.model.hrnet_w48 import HRNetW48Config
from hrnet_tf.model.hrnet_w64 import HRNetW64Config
from hrnet_tf.model.pose_higher_hrnet import pose_higherhrnet, PoseHigherHRNetConfig


SRC_WEIGHTS_DIR = "model_downloads"
DST_WEIGHTS_DIR = "converted_models"


models_and_configs = [
    # (hrnet, HRNetW18SmallV1Config()),
    # (hrnet, HRNetW18SmallV2Config()),
    # (hrnet, HRNetW18Config()),
    # (hrnet, HRNetW18Config()),
    # (hrnet, HRNetW18Config()),
    # (hrnet, HRNetW30Config()),
    (hrnet, HRNetW32Config()),
    # (hrnet, HRNetW40Config()),
    # (hrnet, HRNetW44Config()),
    # (hrnet, HRNetW48Config()),
    # (hrnet, HRNetW48Config()),
    # #### (hrnet, HRNetW48Config()),
    # (hrnet, HRNetW64Config()),
    # (pose_higherhrnet, PoseHigherHRNetConfig()),  # Pose HigherHRNet w32
    # (hrnet, HRNetW32Config()), # Pose HigherHRNet w32, but only the HRNet backbone (without pose head)
]
orig_weights = [
    # "hrnet_w18_small_model_v1.pth",
    # "hrnet_w18_small_model_v2.pth",
    # "hrnetv2_w18_imagenet_pretrained.pth",
    # "HRNet_W18_C_ssld_pretrained.pth",
    # "HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar",
    # "hrnetv2_w30_imagenet_pretrained.pth",
    # "hrnetv2_w32_imagenet_pretrained.pth",
    # "hrnetv2_w40_imagenet_pretrained.pth",
    # "hrnetv2_w44_imagenet_pretrained.pth",
    # "hrnetv2_w48_imagenet_pretrained.pth",
    # "HRNet_W48_C_ssld_pretrained.pth",
    # #### "HRNet_W48_C_cosinelr_cutmix_300epoch.pth.tar", # THIS ONE CANT BE CONVERTED: FILE IS CORRUPTED
    # "hrnetv2_w64_imagenet_pretrained.pth",
    # "pose_higher_hrnet_w32_512.pth",
    # "pose_higher_hrnet_w32_512.pth",
]

output_files = [
    # "hrnet_w18_small_model_v1_imagenet_pretrained_notop.h5",
    # "hrnet_w18_small_model_v2_imagenet_pretrained_notop.h5",
    # "hrnetv2_w18_imagenet_pretrained_notop.h5",
    # "hrnet_w18_ssld_imagenet_pretrained_notop.h5",
    # "hrnet_w18_cosinelr_cutmix_300epoch_imagenet_pretrained_notop.h5",
    # "hrnetv2_w30_imagenet_pretrained_notop.h5",
    "hrnetv2_w32_imagenet_pretrained_notop.h5",
    # "hrnetv2_w40_imagenet_pretrained_notop.h5",
    # "hrnetv2_w44_imagenet_pretrained_notop.h5",
    # "hrnetv2_w48_imagenet_pretrained_notop.h5",
    # "hrnet_w48_ssld_imagenet_pretrained_notop.h5",
    # #### "hrnet_w48_cosinelr_cutmix_300epoch_imagenet_pretrained_notop.h5",
    # "hrnetv2_w64_imagenet_pretrained_notop.h5",
    # "pose_higher_hrnet_w32_512_coco.h5",
    # "pose_higher_hrnet_w32_512_coco_notop.h5",
]

make_multires_output = [
        # [True, False],
        # [True, False],
        # [True, False],
        # [True, False],
        # [True, False],
        # [True, False],
        [True, False],
        # [True, False],
        # [True, False],
        # [True, False],
        # [True, False],
        # #### [True, False],
        # [True, False],
        # [False],
        # [False],
]


if __name__ == '__main__':
    for (model_constr, config), weight_filename, output_path, multires_options in zip(models_and_configs, orig_weights, output_files, make_multires_output):
        weight_path = os.path.join(SRC_WEIGHTS_DIR, weight_filename)
        for multires_output in multires_options:
            if not os.path.exists(DST_WEIGHTS_DIR):
                os.makedirs(DST_WEIGHTS_DIR)
            print(f"Converting {weight_path} with {'MULTI-RESOLUTION output' if multires_output else 'SINGLE-RESOLUTION output'} ...")

            config.MULTI_RESOLUTION_OUTPUT = multires_output
            # Input size is arbitrary, since hrnet backbone as well as pose higher-hrnet are fully convolutional
            model = model_constr(config=config, input_shape=(512, 512, 3))
            include_pose_head = isinstance(config, PoseHigherHRNetConfig)
            load_pytorch_weights(model, weight_file=weight_path,
                                 branches_per_stage=[len(l) for l in config.CHANNELS_PER_STAGE],
                                 modules_per_stage=config.MODULES_PER_STAGE,
                                 blocks_per_stage=config.BLOCKS_PER_STAGE,
                                 multi_resolution_output=config.MULTI_RESOLUTION_OUTPUT,
                                 include_pose_head=include_pose_head,
                                 verbose=False)
            # model.summary()
            out_path = output_path
            if not multires_output and len(multires_options) > 1:
                prefix, ext = os.path.splitext(output_path)
                out_path = prefix + "_single_res" + ext

            out = os.path.join(DST_WEIGHTS_DIR, out_path)
            print("Writing to " + out)
            model.save_weights(out, save_format="h5")

            print("Verifying saved weights ...")
            del(model)
            model = model_constr(config=config, input_shape=(256, 256, 3))
            try:
                model.load_weights(out, by_name=True)
                print("... done")
            except Exception as e:
                print(e)

