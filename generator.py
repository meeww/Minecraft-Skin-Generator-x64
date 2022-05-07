#!C:/Users/joejo/AppData/Local/Programs/Python/Python38/python.exe
import argparse

import sys

from os.path import exists

from ml.pytorch.image_dataset import ImageDataset
from ml.pytorch.wgan.image_wgan import ImageWgan
def init():
    dataset_folder = "data"
    generated_samples_folder = "results"
    discriminator_saved_model = "discriminator.model"
    generator_saved_model = "generator.model"

    image_wgan = ImageWgan(
        image_shape=(4,128,128),
        latent_space_dimension=200,
        use_cuda=True,
        generator_saved_model=generator_saved_model if exists(generator_saved_model) else None,
        discriminator_saved_model=discriminator_saved_model if exists(discriminator_saved_model) else None
    )
    image_wgan.generate(
            sample_folder=generated_samples_folder
    )
    
init()
