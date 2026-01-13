from tqdm import tqdm

from AnimeData import AnimeData
from DCGAN_discriminator import DCGANDiscriminator
from DCGAN_generator import DCGANGenerator
from data_loader import download_data
from hyperparameters import BETA1, NUM_WORKERS, BATCH_SIZE, LATENT_DIM, LEARNING_RATE, BETA2, NUM_EPOCHS, \
    IMAGE_SIZE, sample_dir
from model import train
from utils import make_array, gallery, show_images

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

#download_data()

root = '../data'

from PIL import Image
import glob

stats = (BETA1, BETA1, BETA1), (BETA1, BETA1, BETA1)

image_list = []
rows = []
for filename in glob.glob(root + '/images/*.jpg'):
    im = Image.open(filename)
    rows.append([filename])
    image_list.append(filename)

print(len(image_list))

display_no_images = 10

array = make_array(image_list, display_no_images)
result = gallery(array, display_no_images)
plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.show()

df = pd.DataFrame(rows)
df.to_csv(root + '/data.csv', index=False, header=None)

transform = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Resize(IMAGE_SIZE, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(*stats)])

trainset = AnimeData(root=root + '/data.csv', transform=transform)
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dCGANDiscriminator = DCGANDiscriminator(3).to(device)
dCGANGenerator = DCGANGenerator(LATENT_DIM).to(device)
# random latent tensors
noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)

fake_images = dCGANGenerator(noise)
print(fake_images.shape)

show_images(fake_images, stats)

for real_images in tqdm(trainloader):
    real_images = (real_images).to(device)

show_images(real_images, stats)

loss_fn = torch.nn.MSELoss()

# Create optimizers for the discriminator D and generator G
opt_d = optim.Adam(dCGANDiscriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
opt_g = optim.Adam(dCGANGenerator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

os.makedirs(sample_dir, exist_ok=True)

fixed_latent = torch.randn(IMAGE_SIZE, LATENT_DIM, 1, 1, device=device)

# Complete the training function
losses_g = []
losses_d = []
real_scores = []
fake_scores = []

# Train the GAN
train(dCGANDiscriminator, dCGANGenerator, opt_d, opt_g, stats, loss_fn, trainloader, device, losses_g, losses_d,
      real_scores, fake_scores, fixed_latent, epochs=NUM_EPOCHS)

##Visualize loss curve of D and G
fig, ax = plt.subplots()
plt.plot(losses_g, label='Discriminator', alpha=BETA1)
plt.plot(losses_d, label='Generator', alpha=BETA1)
plt.title("Training Losses")
plt.legend()
