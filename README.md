# kaggle-generative-dog-images
5th place solution for Kaggle's Generative Dog Images competition.

[Kaggle's forum summary post](https://www.kaggle.com/c/generative-dog-images/discussion/104287)

# Description
This solution is based on PyTorch BigGAN implementation by Andy Brock. Some of the original modules were replaced or removed. In case you want to use this model to solve other problems, please, start with the [original repo](https://github.com/ajbrock/BigGAN-PyTorch).

# Hardware
Kaggle's kernel environment with GPU support was used to develop and run this solution. Approximate specs are listed bellow:
- Intel(R) Xeon(R) CPU @ 2.20GHz, 2 cores
- 13 Gb RAM
- 1 x NVIDIA Tesla P100

# Software
- Debian GNU/Linux 9 (stretch)
- Python 3.6.6
- CUDA 10.0.130
- cuddn 7.5.0
- NVIDIA drivers 418.67

# Configuration
The description of available settings can be found in the beginning of *utils.py*. Some of them (like parallel execution or half/mixed precision) won't work, because the required functionality was partially removed.

To change path to model inputs *--data_root* and *--label_root* parameters should be set to appropriate locations.

# Training
To start training a new model execute *run_train.sh*. You may want to edit it first to change some of the settings.

# Generating images
Execute *run_sample.sh*. To set the number of images to be generated change *--sample_num* parameter. Generated images will be stored in the *--samples_root* directory (*./samples* by default). Zip archive with generated images will be saved in the script root directory.

# Pretrained model weights
You can download it from [this Kaggle's dataset](https://www.kaggle.com/dvorobiev/generative-dog-images-biggan-weigths) (Kaggle's account required). Typically the script expects weights to be in *./weights/generative_dog_images* directory, but you can change this behavior by setting *--weights_root* and *--experiment_name* parameters.

# Misc
```
@inproceedings{
brock2018large,
title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
author={Andrew Brock and Jeff Donahue and Karen Simonyan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1xsqj09Fm},
}
```
