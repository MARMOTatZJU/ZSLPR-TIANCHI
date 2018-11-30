# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：data augmentation function
# Author: Yinda XU


import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # iaa.Crop(percent=(0, 0.2)), # random crops
# Small gaussian blur with random sigma between 0 and 0.5.
# But we only blur about 50% of all images.
    # iaa.Sometimes(0.3,
    #     iaa.GaussianBlur(sigma=(0, 0.05))
    # ),
# Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.5, 1.75)),
# Add gaussian noiseself.
# For 50% of all images, we sample the noise once per pixel.
# For the other 50% of all images, we sample the noise per pixel AND
# channel. This can change the color (not only brightness) of the
# pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.2),
# Make some images brighter and some darker.
# In 20% of all cases, we sample the multiplier once per channel,
# which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
# Apply affine transformations to each image.
# Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        # scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(0, 0),
    )
], random_order=True) # apply augmenters in random order

def data_aug_ZSL(image):
    return seq.augment_images(image)


seq_TestAug = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # iaa.Crop(percent=(0, 0.2)), # random crops
# Small gaussian blur with random sigma between 0 and 0.5.
# But we only blur about 50% of all images.
    # iaa.Sometimes(0.3,
    #     iaa.GaussianBlur(sigma=(0, 0.05))
    # ),
# Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.5, 2)),
# Add gaussian noiseself.
# For 50% of all images, we sample the noise once per pixel.
# For the other 50% of all images, we sample the noise per pixel AND
# channel. This can change the color (not only brightness) of the
# pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.2),
# Make some images brighter and some darker.
# In 20% of all cases, we sample the multiplier once per channel,
# which can end up changing the color of the images.
    iaa.Multiply((0.7, 1.3), per_channel=0.2),
# Apply affine transformations to each image.
# Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        # scale={"x": (0.7, 1.0), "y": (0.7, 1.0)},
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-4, 4),
    )
], random_order=True) # apply augmenters in random order

def data_aug_ZSL_TestAug(image):
    return seq_TestAug.augment_images(image)
