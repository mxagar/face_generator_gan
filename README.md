# Face Generation with a Convolutional Generative Adversarial Network (GAN)

This repository contains a Convolutional Generative Adversarial Network (GAN) that generates faces based on the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A subset of the dataset is used, which can be downloaded [from here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip). The subset, which consists of 89,931 images, has been processed to contain only `64 x 64 x 3` cropped faces without any annotations. Here's a sample of 16 images:

<p align="center">
  <img src="./assets/face_inputs.jpg" alt="Some processed samples from the CelebA faces dataset.">
</p>

The project is implemented with [Pytorch](https://pytorch.org/) and it uses materials from the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101), which can be obtained in their original (non-implemented) form in [project-face-generation](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-face-generation).

Regarding the *current* results, I would say that they look like faces :sweat_smile:

<p align="center">
  <img src="./assets/face_outputs_epoch_50.jpg" alt="Some generated samples in the epoch 50.">
</p>

... but most of them would fail a human-level classification on whether they're real. However, I find it very remarkable that such a simple network with untuned hyperparameters from the literature and only 1.5h of GPU training is able to yield such face images. The section [Improvements](#improvements-and-next-steps) details how I will approach the next steps &mdash; when I find time for that :sweat_smile:.

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

In order to use the model, you need to install the [dependencies](#dependencies) and execute the notebook [`dl_face_generation.ipynb`](dl_face_generation.ipynb), which is the main application file that defines and trains the network.

Next, I give a more detailed description on the contents and the usage.

### Overview of Files and Contents

Altogether, the project directory contains the following files and folders:

```
.
????????? Instructions.md                     # Original project requirements
????????? README.md                           # This file
????????? assets/                             # Images used in the notebook
????????? dl_face_generation.ipynb            # Implementation notebook
????????? problem_unittests.py                # Unit tests
????????? requirements.txt                    # Dependencies
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

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [`requirements.txt`](requirements.txt) file.

A short summary of commands required to have all in place is the following:

```bash
conda create -n face-gan python=3.6
conda activate face-gan
conda install pytorch -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Brief Notes on Generative Adversarial Networks (GAN) and the Chosen Architecture

Generative Adversarial Networks (GAN) were introduced by [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661) and they strengthened a very promising new research branch which focuses on creating unseen data samples (e.g., images) from a learned latent space. [Radford et al., 2016](https://arxiv.org/abs/1511.06434v2) presented Deep Convolutional GANs (DCGANs), which can cope with complex images when deep. The main difference is the use of convolutions and transpose convolutions instead of fully connected layers.

A GAN consists of two networks that compete with each other:

- The **Generator** `G()` network learns to generate fake but realistic/meaningful data from a `z` noise input: `x = G(z)`.
- The **Discriminator** `D()` network learns to distinguish between fake and real data; its output is 1 (real) or 0 (fake): `D(x) -> (0, 1)`.

Let's consider that our data is formed by images, `x`. In that case:

- The Generator upsamples a fix-sized random vector to an image `x = G(z)`, with a given size. [Transpose convolutions](https://en.wikipedia.org/wiki/Deconvolution) are used to that end.
- The Discriminator performs a classification task, which determines whether the input image is real or false. Strided convolutions are used for that task, i.e., the image size is successively halvened using the appropriate kernel size and stride instead of pooling.

<p align="center">
  <img src="./assets/GANs.png" alt="GAN Architecture" width=800px>
</p>


The training occurs in two phases, which are executed one after the other often for every batch of real images. In the **first phase**, the Discriminator is trained: first real images with a label value 1 (i.e., *real*) are used, then, images generated by the Generator are used with label value 0 (i.e., *fake*). Those generated images come from a batch of random noise vectors `z` passed to `G()`.

In the **second phase**, the Generator is trained to fool the Discriminator: we produce more fake images `x = G(z)` and feed them to the Discriminator, `D(G(z))`, but labelled with 1, that is, as if they were real! It is as if the Generator were a counterfeiter trying to trick the police, i.e., the Discriminator. The loss of the deceived Discrimanator is used in the optimizer of the Generator, i.e., we update the weights of the Generator with the cost originated from using fake images and flipped labels.

The result is the following:

- The Generator never sees a real image, but through the gradients of the Discriminator loss, it becomes very good at creating fake images.
- The Discriminator tends to produce an outcome close to 0.5 with time, because it's not able to distinguish between real and fake images.
- The random vector `z` represents a learned latent space which contains all possible variations of the features in the images we have shown to the Discriminator; i.e., if we've used faces, `z` is the compressed space of all faces &mdash; not the faces we have used, but *all* possible unseen faces given the features captured from the dataset.

Many GAN derivate architectures use this approach, creating many interesting applications; for instance:

- Pix2Pix, presented by [Isola et al.](https://arxiv.org/pdf/1611.07004.pdf), learns to map images from different domains (e.g., sketches and real images, or segmentations and real images). The dataset is composed by paired images from the learned domains and its major characteristic resides in the Generator, which has an encoder-decoder architecture: the encoder compresses the images from one domain to a `z` vector equivalent to the noise in a regular GAN; the decoder is the typical Generator. Thus, the Pix2Pix Generator is similar to an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder).
- CyclGAN, presented by [Zhu et al.](https://arxiv.org/pdf/1703.10593.pdf), learns to map characteristics of a domain to another using unpaired images, e.g., we can convert the horse in an image into a zebra or we can transform a winter landscape into a summer view. The setup is more complex, and it consists of two Generators and two Discriminators which are trained together; the Generators have also an architecture hat resembles an autoencoder and they map images from domain `X` to `Y`, and vice versa. Meanwhile, there is a Discriminator for each mapping.

## Practical Implementation Notes

This small project implements a Deep Convolutional GAN (DCGAN), following the guidelines presented in [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661), [Radford et al., 2016](https://arxiv.org/abs/1511.06434v2), and [Salimans et al., 2016](https://arxiv.org/abs/1606.03498). The following non-exhaustive list collects some notes related to the implementation:

- [Leaky ReLU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7) is used in the Discriminator to facilitate the propagation of the gradient in the entire network, since its gradient is not `0` for negative values.
- Smooth real loss is implemented, i.e., labels with value `1` are scaled with `0.9` so that they are a little bit below their value. It regularizes the system and decreases extreme predictions.
- Strided convolutions are used in the Discriminator instead of pooling to halven the image size. The output image width `W_out` in a convolution layer is given by the [formula](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) `W_out = (W_in + 2P - K)/S + 1`, with `P` padding, `K` kernel size, and `S` stride. Defining `P = 1, K = 4, S = 2`, we get `W_out = W_in/2`.
- Batch normalization is used in all the layers except in the output layer of the Generator and in the input layer of the Discriminator.
- No sigmoid activation is used in the Discriminator; instead, logits are output and the loss function is [`BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), which combines a sigmoid with a `BCELoss()`. This approach is more stable.
- The Discriminator is a sequence of 4 convolutional layers (each with batch normalization and leaky ReLU) and two linear layers with dropout in-between. The depth in the 1st convolution is `32` and it doubles every layer.
- The Generator is equivalent to the Discriminator, but it upscales a random vector of size `z_size = 256` to an image of size `32 x 32 x 3`. Four transpose convolutions are used (each with batch normalization and ReLU) in sequence and the output is activated with `tanh()`.

## Improvements and Next Steps

The current results are far from high quality faces such as in [StyleGAN 3, 2021](https://arxiv.org/abs/2106.12423); however, as mentioned in the introduction, I think it's remarkable that the defined simple model with untuned hyperparameters from the literature is able to generate those faces with only 1.5h of GPU training.

In case I have time, I will try to improve the results applying these items:

- [ ] Try the [Wasserstein distance](https://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them#wassGANs) in the loss function, since it correlates better with the image quality. That way, it's more clear when we should stop training.
- [ ] Increase the image size to `64 x 64` to create bigger images.
- [ ] Increase the number of channels (e.g., 2x or 4x), controlled with `conv_dim`.
- [ ] Padding and normalization techniques can achieve better qualities? Check literature.
- [ ] Vary the learning rate during training; example: [Original CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- [ ] Try other weight initializations, e.g., the [Xavier Initialization](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/).
- [ ] Currently, label smoothing is implemented but deactivated; activate it.
- [ ] Read thoroughly and the techniques commented in the paper [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498).
- [ ] I have barely varied the hyperparameters! I should systematically search in the paramater space; e.g.:
  - [ ] The number of epochs.
  - [ ] The `batch_size`; check [this Standford / Karpathy article](https://cs231n.github.io/neural-networks-3/#hyper).
  - [ ] The `beta` values for the optimizer; check [this article by Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/index.html#adam).
  - [ ] The size of the random/noise vector `z_size`.
  - [ ] The slope of the leaky ReLU.
  - [ ] etc.
- [ ] Address the bias in the dataset: many white, blonde and female faces are generated.
  - [ ] Re-sample the training dataset?
  - [ ] Use another dataset?
- [ ] Try architectures like [PairedCycleGAN](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf), in which face properties are mapped from one domain to another. I have implemented an example of a CycleGAN in which winter and summer domain image properties are mapped from one set to another: [CycleGAN Summer/Winter](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/cycle-gan).


## Interesting Links

- [My notes and code](https://github.com/mxagar/computer_vision_udacity) on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).
- My toy project on [image captioning](https://github.com/mxagar/image_captioning).
- [iGAN: Interactive GANs](https://github.com/junyanz/iGAN)
- Pix2Pix paper: [Image-to-Image Translation with Conditional Adversarial Networks, Isola et al.](https://arxiv.org/pdf/1611.07004.pdf)
- [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
- [Image to Image demo](https://affinelayer.com/pixsrv/)
- CycleGAN paper: [Unpaired Image-to-Image Translationusing Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
- [CyCleGAN & Pix2Pix Repo with Pytorch (by author Zhu)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [DCGAN Tutorial, Pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Cool GAN Applications](https://jonathan-hui.medium.com/gan-some-cool-applications-of-gans-4c9ecca35900)
- [Compilation of Cool GAN Projects](https://github.com/nashory/gans-awesome-applications).
- GAN paper: [Generative Adversarial Networks, 2014](https://arxiv.org/abs/1406.2661).
- [Improved Techniques for Training GANs, 2016](https://arxiv.org/abs/1606.03498).
- [DCGAN Paper, 2016](https://arxiv.org/abs/1511.06434v2).
- [Understanding the backward pass through Batch Normalization Layer](http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
- [Different ReLU Methods](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7)

**Very realistic face generation**:

- [StyleGAN, 2019](https://arxiv.org/abs/1710.10196/).
- [StyleGAN 2, 2020](https://paperswithcode.com/method/stylegan2).
- [StyleGAN 3, 2021](https://arxiv.org/abs/2106.12423).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.
