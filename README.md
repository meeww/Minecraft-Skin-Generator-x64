## Minecraft Skin Generator
This project uses Artificial Intelligence (AI) to "dream up" Minecraft skins. It is trained on existing Minecraft skins to "understand" what they look like so it can create its own.

## Installation
Please install Python 3.8 before continuing. https://www.python.org/downloads/release/python-380/

Create a directory to store the generator and enter that directory.
   
     cd path/to/directory
     git clone https://github.com/meeww/Minecraft-Skin-Generator-x64.git
     pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

## Training Data
 download the minecraft skin image dataset, unzip its contents and rename the folder containing the skins to data
 then upload the folder into the directory where you cloned the generator
 https://www.reddit.com/r/datasets/comments/cmccb8/minecraft_skins_image_dataset/
    
## Training the model
The model can continue training by using the command:

    $ python3 main.py --mode train --dataset data --epochs 5

The models will be saved to ./discriminator.model and ./generator.model
   
## Generating new skins
You can generate new skins from the model by running generate.py. The output will be 4 skins with a padding of 2px. So if you want to seperate the textures also run crop.py

    $ python3 generate.py
    $ python3 crop.py

Check for the generated skins at `./results/*.png`.

## Credit   

The GAN used to generate images is based heavily on [eriklindernoren's WGAN implementation](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py).
