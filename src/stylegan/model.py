import os

import torch
from matplotlib import pyplot as plt
from torch import optim, nn

from stylegan.StyleGAN_discriminator import StyleGANDiscriminator
from stylegan.StyleGAN_generator import StyleGANGenerator
from stylegan.hyperparameters import BETA1, BETA2, LEARNING_RATE
from stylegan.utils import save_samples


def train_stylegan(device, trainloader, fixed_latent, num_epochs=1,
                   batch_size=32, latent_dim=512, lr=LEARNING_RATE):

    # Initialize models
    styleGANGenerator = StyleGANGenerator(latent_dim=latent_dim).to(device)
    styleGANDiscriminator = StyleGANDiscriminator().to(device)

    start_idx=1

    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    styleGANGenerator.apply(weights_init)
    styleGANDiscriminator.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(styleGANGenerator.parameters(), lr=lr, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(styleGANDiscriminator.parameters(), lr=lr, betas=(BETA1, BETA2))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, device=device)

    # Training history
    g_losses = []
    d_losses = []

    print("Starting StyleGAN training...")

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, real_imgs in enumerate(pbar):
            batch_size_curr = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels
            real_labels = torch.ones(batch_size_curr, device=device)
            fake_labels = torch.zeros(batch_size_curr, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            real_output = styleGANDiscriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = styleGANGenerator(z)
            fake_output = styleGANDiscriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = styleGANGenerator(z)
            fake_output = styleGANDiscriminator(fake_imgs)

            # Generator loss
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Update progress bar
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix({'G_loss': g_loss.item(), 'D_loss': d_loss.item()})



        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(trainloader)
        avg_d_loss = epoch_d_loss / len(trainloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")

        # Save generated images
        save_samples(epoch+start_idx, fixed_latent,styleGANGenerator,'styleGANGenerator', show=True)

        state_dis = {'styleGANDiscriminator_model': styleGANDiscriminator.state_dict(), 'epoch': epoch}
        state_gen = {'styleGANGenerator_model': styleGANGenerator.state_dict(), 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state_dis, 'checkpoint/'+'styleGANDiscriminator'+str(epoch+1)) #each epoch
        torch.save(state_gen, 'checkpoint/'+'styleGANGenerator__'+str(epoch+1)) #each epoch

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('StyleGAN Training Losses')
    plt.close()

    print("Training complete!")
    return styleGANGenerator, styleGANDiscriminator
