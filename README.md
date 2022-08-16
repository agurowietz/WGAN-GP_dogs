# WGAN-GP_dogs
Image generation of dogs

## Objective
The aim of this work comprises the generation of images showing the author’s Chihuahuas using a WGAN with gradient penalty. Due to the limited amount of suitable training data, the approach of utilizing a pre-trained dog WGAN-GP was chosen. First, the WGAN was trained on a dataset containing 140,864 dog images of different breeds and the pre-trained critic and generator were then further used for training with the smaller dataset of desired Chihuahua pictures in order improve the quality of generated Chihuahua images.

## Datasets
- data/chihuahua
Contains ca. 1300 images of the author's Chihuahuas
- data/tsinghua_dogs
Contains ca. 140,000 dog images from various breeds taken from:         
A new dataset of dog breed images and a benchmark for fine-grained classification, Zou, Ding-Nan and Zhang, Song-Hai and Mu, Tai-Jiang and Zhang, Min, Computational Visual Media, 2020, https://doi.org/10.1007/s41095-020-0184-6

## Training
- run src/wgan-gp.py to start training vanilla WGAN-GP on Chihuahua dataset
- run src/wgan-gp_pretrained.py to start training on Chihuahua dataset with pre-trained WGAN-GP

## Output
- generates an image of an output batch of the generator and loss plot every 10 epochs under /output
- saves models weights every 10 epochs under /model
