# Satellite SuperResolution using GAN / 4x Improvement

### Model Training and Architecture

This a slightly modified version of `SRGAN` model as it does not have the `BatchNormalization` layer as mentioned in the original paper. The training happens in two stages. The data preprocessing and metrics have been modified to support satellite reflectance data values (0-1) instead of traditional 0-255 range.

The model supports only `2x`, `3x` and `4x` upscaling of TIF Images however it uses feature maps of VGG network in the final stage

### Results

**AGRI**
![AGRI Image](results/agri.PNG)

**URBAN**
![Urbabn Image](results/urban.PNG)


1. **PSNR Training**

First very deep ResNet architecture using the concept of GANs to form a perceptual loss function as measured by PSNR and structural similarity (SSIM) with our 16 blocks deep ResNet (SRResNet) optimized for MSE.

2. **GAN Training**

Traditional SRGAN model is used with `L1 Loss` for training. The backend used is VGG-19

![GAN Image](gan.png)

### Data

[Sample data is available here](https://drive.google.com/file/d/1LGugOJuX1jec_3cnnDNDItBdJhi-jK4R/view?usp=sharing). Standard `TIF` imagery (reflectance) is used for model building. The zip file should be extracted into `data/` directory

### Installation

Separate conda environment with `Python >= 3.7` and `tensorflow-gpu` is recommended. 
`python setup.py install`

### Running Training

1. Configure File: `options\train\SRResNet_SRGAN\train_MSRResNet_x4.yml`
2. Configure File: `options\train\SRResNet_SRGAN\train_MSRGAN_x4.yml`

### Running Inference

1. Configure File: `options\test\SRResNet_SRGAN\test_MSRGAN_x4.yml`

### Starting the process

1. Running the SRRSNET Model `python basicsr/train.py --options options\train\SRResNet_SRGAN\train_MSRResNet_x4.yml`
2. Running the SRGAN Model `python basicsr/train.py --options options\train\SRResNet_SRGAN\train_MSRGAN_x4.yml`
3. Inference from SRGAN Model `python basicsr/test.py --options options\test\SRResNet_SRGAN\test_MSRGAN_x4.yml`

## :heart: References

Inspired from `https://github.com/xinntao/BasicSR`