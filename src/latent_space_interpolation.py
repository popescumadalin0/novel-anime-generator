import torch

from hyperparameters import LATENT_DIM
from stylegan import styleGANGenerator


def generate_samples(generator, num_samples=16, latent_dim=512):
    """Generate samples from trained StyleGAN"""
    device = next(generator.parameters()).device
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_imgs = generator(z)
        print(fake_imgs)


def latent_space_interpolation(generator, num_steps=10, latent_dim=512):
    """
    Create smooth interpolation between two random points in latent space
    """
    device = next(generator.parameters()).device
    generator.eval()

    with torch.no_grad():
        # Generate two random latent codes
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)

        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_images = []

        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z)
            interpolated_images.append(img)

        # Save grid
        grid = torch.cat(interpolated_images, dim=0)
        print(grid)


print("\nGenerating samples...")
generate_samples(styleGANGenerator, num_samples=16, latent_dim=LATENT_DIM)

# Latent space interpolation
print("\nCreating latent space interpolation...")
latent_space_interpolation(styleGANGenerator, num_steps=10, latent_dim=LATENT_DIM)
