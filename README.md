# Face Generation with a Convolutional Generative Adversarial Network (GAN)

This repository contains a Convolutional Generative Adversarial Network (GAN) that generates faces based on the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A subset of the dataset is used, which can be downloaded [from here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip). The subset, which consists of 89,931 images, has been processed to contain only of `64 x 64 x 3` cropped faces without any annotations. Here's a sample of 14 images:

<p align="center">
  <img src="./assets/processed_face_data.png" alt="Some processed samples from the CelebA faces dataset.">
</p>

The project is implemented with [Pytorch](https://pytorch.org/) and it uses materials from the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891), which can be obtained in their original (non-implemented) form in [project-face-generation](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-face-generation).

:construction: Regarding the results, ...

... and that's with very few hours of effort and GPU training, so I I'd say it's a good starting point :sweat_smile:

Table of Contents:

- [Face Generation with a Convolutional Generative Adversarial Network (GAN)](#face-generation-with-a-convolutional-generative-adversarial-network-gan)
  - [How to Use This](#how-to-use-this)
    - [Overview of Files and Contents](#overview-of-files-and-contents)
    - [Dependencies](#dependencies)
  - [Brief Notes on Generative Adversarial Networks (GAN) and the Chosen Architecture](#brief-notes-on-generative-adversarial-networks-gan-and-the-chosen-architecture)
  - [Practical Implementation Notes](#practical-implementation-notes)
  - [Improvements and Next Steps](#improvements-and-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## How to Use This

In order to use the model, you need to install the [dependencies](#dependencies) and execute the notebook [`dlnd_face_generation.ipynb`](dlnd_face_generation.ipynb), which is the main application file that defines and trains the network.

Next, I give a more detailed description on the contents and the usage.

### Overview of Files and Contents

Altogether, the project directory contains the following files and folders:

```
.
├── Instructions.md                     # Original project requirements
├── README.md                           # This file
├── assets/                             # Images used in the notebook
├── dl_face_generation.ipynb            # Implementation notebook
├── problem_unittests.py                # Unit tests
└── requirements.txt                    # Dependencies
```

As already introduced, the notebook [`dl_face_generation.ipynb`](dl_face_generation.ipynb) takes care of almost everything. The file [`problem_unittests.py`](problem_unittests.py) is used to check different parts of the notebook.

All in all, the following tasks are implemented in the project notebook:

- A data loader is defined, which is able to deliver `batch_size` number of images resized to `img_size`.
- An image scaling function is defined to map pixel values to `[-1,1]`.
- The Discriminator and the Generator models are defined with convolution and transpose convolution layers. Instead of pooling, strided convolutions are used, and batch normalization and leaky ReLU are are also applied, as in the original [DCGAN Paper](https://arxiv.org/abs/1511.06434v2).
- The models are instantiated and initialized with custom weights: `N(0, 0.02)`.
- The loss functions are defined: one for the real images and another one for the fake images produced by the Generator.
- The optimizers are defined using values from the [DCGAN Paper](https://arxiv.org/abs/1511.06434v2): one for the Discriminator and another one for the Generator.
- The typical GAN training loop is defined and executed.
- Learning curves (i.e., loss values) are plotted.
- Generated images from a fixed set of noise vectors is displayed for given epochs.

When the notebook is executed, several other artifacts are generated:

- A folder with the dataset: `processed_celeba_small/`
- A folder for the latest model saved after every epoch: `checkpoints_gan/`
- A file which contains the images generated from a fixed set of noise vectors every epoch: `train_samples.pkl`.

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file.

A short summary of commands required to have all in place is the following:

```bash
conda create -n text-gen python=3.6
conda activate text-gen
conda install pytorch -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Brief Notes on Generative Adversarial Networks (GAN) and the Chosen Architecture

:construction:

TBD.

## Practical Implementation Notes

:construction:

TBD.

## Improvements and Next Steps

:construction:

TBD.

- [ ] A
- [ ] B

## Interesting Links

- [My notes and code](https://github.com/mxagar/computer_vision_udacity) on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).
- My toy project on [image captioning](https://github.com/mxagar/image_captioning).
- [iGAN: Interactive GANs](https://github.com/junyanz/iGAN)
- Pix2Pix paper: [Image-to-Image Translation with Conditional Adversarial Networks, Isola et al.](https://arxiv.org/pdf/1611.07004.pdf)
- [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
- [Image to Image demo](https://affinelayer.com/pixsrv/)
- CycleGAN paper: [Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
- [CyCleGAN & Pix2Pix Repo with Pytorch (by author Zhu)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [DCGAN Tutorial, Pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Cool GAN Applications](https://jonathan-hui.medium.com/gan-some-cool-applications-of-gans-4c9ecca35900)
- [Compilation of Cool GAN Projects](https://github.com/nashory/gans-awesome-applications).
- GAN paper: [Generative Adversarial Networks, 2014](https://arxiv.org/abs/1406.2661).
- [Improved Techniques for Training GANs, 2016](https://arxiv.org/abs/1606.03498).
- [DCGAN Paper, 2016](https://arxiv.org/abs/1511.06434v2).

**Very realistic face generation**:

- [StyleGAN, 2019](https://arxiv.org/abs/1710.10196/).
- [StyleGAN 2, 2020](https://paperswithcode.com/method/stylegan2).
- [StyleGAN 3, 2021](https://nvlabs.github.io/stylegan3/).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.
