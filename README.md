# hrnet-tensorflow

Tensorflow (Keras) implementation of various **HRNet** and **HigherHRNet** variants that are fully compatible with the original pytorch implementations. Includes **weight conversion tool** for the original authors' pretrained pytorch models.

The configurations of all base HRNet variants and their ImageNet pre-trained weights come from the authors' [HRNet GitHub repo](https://github.com/HRNet/HRNet-Image-Classification).

HigherHRNet configuration for 2D human pose estimation and a COCO-trained model comes from the authors' [HigherHRNet GitHub repo](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation).

**Note:** We do not provide the converted model weights for download. You have to convert them yourself from the original pytorch-based weights from the respective repos.

## Requirements

  * `Python >= 3.8`
  * `tensorflow >= 2.4.2`
  * `torch >= 1.7.1` (Only for one-time weight conversion, if required)

## Ussage Example

```python
import tensorflow as tf
from hrnet_tf.model.hrnet_w32 import hrnet_w32


input_shape = (512, 512, 3)
# Path to converted weights file (Keras .h5 format)
weights = "converted_models/hrnetv2_w32_imagenet_pretrained_notop.h5"

# Create HRNet-w32 model and load weights
model = hrnet_w32(input_shape=input_shape, weights=weights)

# Create dummy input data and run a forward pass through the model
# Use standard imagenet normalization (mean and std) for real inputs!
dummy_input = tf.random.normal((1, *input_shape))
# Run the model, output consists of one tensor for each resolution branch
dummy_outputs = model(dummy_input)
# Select the output of the highest-resolution branch (i.e. the first one)
highest_res_output = dummy_outputs[0]
print("Output shape:", highest_res_output.shape)
```

## Pre-configured Model Variants

### HRNet
We provide pre-configured common HRNet variants (w18 up until w64), but you can easily configure a custom variant yourself.
The weight conversion tool covers the model variants / pretrained weights below. Consult the original authors' [repo](https://github.com/HRNet/HRNet-Image-Classification) for download links to the weights.

| model |#Params | GFLOPs |
| --- | ---- | ---- |
| HRNet-W18-C-Small-v1 | 13.2M | 1.49 |
| HRNet-W18-C-Small-v2 | 15.6M | 2.42 |
| HRNet-W18-C | 21.3M | 3.99 | 23.2% |
| HRNet-W30-C | 37.7M | 7.55 | 21.8% |
| HRNet-W32-C | 41.2M | 8.31 | 21.5% |
| HRNet-W40-C | 57.6M | 11.8 | 21.1% |
| HRNet-W44-C | 67.1M | 13.9 | 21.1% |
| HRNet-W48-C | 77.5M | 16.1 | 20.7% |
| HRNet-W64-C | 128.1M | 26.9 | 20.5% |
| HRNet-W18-C (w/ CosineLR + CutMix + 300epochs) | 21.3M | 3.99 |
| ~~HRNet-W48-C (w/ CosineLR + CutMix + 300epochs)~~ Not convertable due to file corruption in original weights...  | 77.5M | 16.1 |
| HRNet-W18-C-ssld (converted from PaddlePaddle) | 21.3M | 3.99 |
| HRNet-W48-C-ssld (converted from PaddlePaddle) | 77.5M | 16.1 |

### Pose HigherHRNet

We provide a HigherHRNet-w32 configuration for 2D human pose estimation, which matches the respective model variant from the original authors' [repo](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). You can find the COCO-pretrained model (for weight conversion) and evaluation results there.

| Method             | Backbone | Input size | #Params | GFLOPs |
|--------------------|----------|------------|---------|--------|
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   |

## Weight Conversion Guide pytorch -> tensorflow (.h5)

To utilize the weight conversion tool (`convert_weights.py`), please follow these steps:

1. Weight conversion requires the additional installation of PyTorch. See the optional dependency on `torch`, as specified in `requirements.txt`.
2. Download the original pre-trained models from their respective repositories (see model list above) and place them into the `model_downloads` directory (`*.path` or `*.path.tar` files).
3. Adjust the selection of models within the script to your set of models/weights that you want to convert.
4. Run the conversion script:
   ```bash
   python convert_weights.py
   ```
5. Successfully converted TensorFlow weights will be placed as `*.h5` format in the `converted_models` directory. By default, we generate two sets of weights from each HRNet (not HigherHRNet!) model: `[...].h5` with all four output tensors from the four resolution streams (`MULTI_RESOLUTION_OUTPUT = True`), and `[...]_single_res.h5` where only the single output tensor from the highest resolution branch is computed (`MULTI_RESOLUTION_OUTPUT = False`)


## Citation
If you find this work or code is helpful in your research, please cite the original authors:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
````
