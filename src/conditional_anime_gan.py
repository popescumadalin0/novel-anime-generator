"""
Conditional DCGAN for Anime Face Generation
Allows conditioning on attributes like hair color or eye color
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image

# Hyperparameters
LATENT_DIM = 100
NUM_CLASSES = 10  # Number of different attributes (e.g., hair colors)
EMBEDDING_DIM = 50
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConditionalGenerator(nn.Module):
    """Conditional DCGAN Generator with class embedding"""

    def __init__(self, latent_dim=100, num_classes=10, embedding_dim=50, feature_maps=64):
        super(ConditionalGenerator, self).__init__()

        # Embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Combined input dimension
        input_dim = latent_dim + embedding_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), -1, 1, 1)

        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_embedding], dim=1)

        return self.main(gen_input)


class ConditionalDiscriminator(nn.Module):
    """Conditional DCGAN Discriminator with class embedding"""

    def __init__(self, num_classes=10, embedding_dim=50, feature_maps=64):
        super(ConditionalDiscriminator, self).__init__()

        # Embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # First layer processes image + embedded label
        self.initial = nn.Sequential(
            nn.Conv2d(3, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Label processing
        self.label_proj = nn.Sequential(
            nn.Linear(embedding_dim, IMAGE_SIZE * IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Main discriminator network
        self.main = nn.Sequential(
            nn.Conv2d(feature_maps + 1, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        # Process image
        x = self.initial(images)

        # Process label and reshape to match spatial dimensions
        label_embedding = self.label_embedding(labels)
        label_map = self.label_proj(label_embedding)
        label_map = label_map.view(label_map.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)

        # Downsample label map to match x spatial dimensions
        label_map = nn.functional.interpolate(label_map, size=(x.size(2), x.size(3)))

        # Concatenate along channel dimension
        x = torch.cat([x, label_map], dim=1)

        return self.main(x).view(-1, 1)


def generate_conditional_images(generator, num_classes=10, samples_per_class=5, output_path='conditional_output.png'):
    """Generate images for each class/attribute"""
    generator.eval()

    all_images = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            noise = torch.randn(samples_per_class, LATENT_DIM, 1, 1, device=device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)

            fake_images = generator(noise, labels)
            all_images.append(fake_images)

    all_images = torch.cat(all_images, dim=0)
    save_image(all_images, output_path, normalize=True, nrow=samples_per_class)

    print(f"Generated conditional images saved to {output_path}")
    print(f"Rows represent different classes/attributes (0-{num_classes - 1})")


if __name__ == "__main__":
    print("=" * 70)
    print("CONDITIONAL ANIME FACE GENERATION")
    print("=" * 70)
    print("\nThis is an extension that allows conditioning on attributes.")
    print("Note: Requires labeled dataset with attributes (hair color, eye color, etc.)")
    print("\nFor full training, you'll need to:")
    print("  1. Prepare a labeled dataset")
    print("  2. Implement the training loop (similar to anime_dcgan.py)")
    print("  3. Condition generation on specific attributes")
