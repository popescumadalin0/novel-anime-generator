import os

from matplotlib import pyplot as plt
from torchvision.utils import save_image, make_grid

from dcgan.utils import denorm
from stylegan.hyperparameters import sample_dir


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