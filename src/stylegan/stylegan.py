import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dcgan.AnimeData import AnimeData
from hyperparameters import LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, LATENT_DIM, BETA1, IMAGE_SIZE, NUM_WORKERS
from model import train_stylegan

root = '../data'

stats = (BETA1, BETA1, BETA1), (BETA1, BETA1, BETA1)

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


fixed_latent = torch.randn(IMAGE_SIZE, LATENT_DIM, 1, 1, device=device)

styleGANGenerator, styleGANDiscriminator = train_stylegan(
    device=device,
    trainloader=trainloader,
    fixed_latent=fixed_latent,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    latent_dim=LATENT_DIM,
    lr=LEARNING_RATE
)
