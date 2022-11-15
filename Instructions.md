# Face Generation with a Convolutional Generative Adversarial Network (GAN): Instructions

The original project files can be found in the following repository: [udacity/deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch).

## Requirements

Summary of the Udacity [rubric](https://review.udacity.com/#!/rubrics/2261/view), check upon submission:

- Submit (1) unit test script + (2) notebook + (3) its HTML version in a ZIP file.
- Notebook passes all unit tests.
- `get_dataloader` correctly implemented.
- `scale` correctly implemented.
- Discriminator correctly implemented.
- Generator correctly implemented.
- Weight initialization correctly implemented.
- Real and fake loss functions correctly implemented.
- Correct optimizers.
- Training loop correctly implemented.
- Hyperparameters correctly chosen; deep enough models, convergence in optimization.
- Realistic faces are generated.
- Answer the question about how we could improve our model.

The requirements of the specific parts are written in the notebook itself.

## Improvement Suggestions

- Create a deeper model which generates larger images, e.g., 128x128.
- Read literature: is it possible to use padding and normalization to generate highres images?
- Implement a learning rate that changes over time. Example in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Extend the model to a CycleGAN, e.g. like in [Style Transfer in Faces with CycleGAN](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf).