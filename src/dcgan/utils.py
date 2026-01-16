import os
import random

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

from hyperparameters import IMAGE_SIZE, sample_dir


def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def make_array(image_list, display_no_images):

    arr = []
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    
    for i in range(display_no_images):
        random_image_path = random.choice(image_list)
        
        img = Image.open(random_image_path).convert('RGB')
        img_resized = img.resize(target_size) 
        arr.append(np.asarray(img_resized))
    return np.array(arr)


def gallery(array, ncols=8):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def show_images(images, stats, nmax=IMAGE_SIZE):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach().cpu()[:nmax], stats), nrow=8).permute(1, 2, 0))

def save_samples(index, latent_tensors, generator, generatorName, stats, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-{generatorName}-images-{0:0=4d}.png'.format(index, generatorName=generatorName)
    save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()