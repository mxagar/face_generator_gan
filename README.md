# Face Generation with a Convolutional Generative Adversarial Network (GAN)

This repository contains a Convolutional GAN which generates faces based on the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

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
├── dlnd_face_generation.ipynb          # Implementation notebook
├── problem_unittests.py                # Unit tests
└── requirements.txt                    # Dependencies
```

As already introduced, the notebook [`dlnd_face_generation.ipynb`](dlnd_face_generation.ipynb) takes care of almost everything. The file [`problem_unittests.py`](problem_unittests.py) is used to check different parts of the notebook.

All in all, the following sections/tasks are implemented in the project notebook:

:construction:

- A
- B
- ...

When the notebook is executed, several other artifacts are generated:

- A folder with the dataset: `processed_celeba_small/`
- ...

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

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.
