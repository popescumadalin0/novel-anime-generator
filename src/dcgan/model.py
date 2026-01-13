import os

import torch
from tqdm.notebook import tqdm

from optimizers import Real_loss, Fake_loss
from hyperparameters import BATCH_SIZE, LATENT_DIM
from utils import save_samples


def train(dCGANDiscriminator, dCGANGenerator, d_optimizer, g_optimizer, stats,loss_fn, trainloader, device, losses_g, losses_d, real_scores, fake_scores, fixed_latent, epochs=1):
    iter_count = 0
    start_idx = 1
    for epoch in range(epochs):
        for real_images in tqdm(trainloader):
            real_images = real_images.to(device)
            # Pass real images through discriminator
            D_out_real = dCGANDiscriminator(real_images)
            label_real = torch.full(D_out_real.shape, 1.0).to(torch.device(device))
            real_loss = Real_loss(label_real, D_out_real, device, loss_fn)
            real_score = torch.mean(D_out_real).item()

            # Generate fake images
            noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(torch.device(device))
            fake_images = dCGANGenerator(noise)

            # Pass fake images through discriminator
            D_out_fake = dCGANDiscriminator(fake_images)
            label_fake = torch.full(D_out_fake.shape, 0).to(torch.device(device))
            fake_loss = Fake_loss(label_fake, D_out_fake, device, loss_fn)
            fake_score = torch.mean(D_out_fake).item()

            # Update discriminator weights
            loss_d = real_loss + fake_loss

            d_optimizer.zero_grad()
            loss_d.backward(retain_graph=True)
            d_optimizer.step()

            # Generate fake images
            noise2 = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(torch.device(device))
            fake_images2 = dCGANGenerator(noise2)

            gen_steps = 1
            for i in range(0, gen_steps):
                # Try to fool the discriminator
                D_out_fake2 = dCGANDiscriminator(fake_images2)

                # The label is set to 1(real-like) to fool the discriminator
                label_real1 = torch.full(D_out_fake2.shape, 1.0).to(torch.device(device))
                loss_g = Real_loss(label_real1, D_out_fake2, device, loss_fn)

                # Update generator weights
                g_optimizer.zero_grad()
                loss_g.backward(retain_graph=(i < gen_steps - 1))
                g_optimizer.step()

        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        # Save generated images
        save_samples(epoch + start_idx, fixed_latent, dCGANGenerator, 'dCGANGenerator', stats, show=True)

        state_dis = {'dCGANDiscriminator_model': dCGANDiscriminator.state_dict(), 'epoch': epoch}
        state_gen = {'dCGANGenerator_model': dCGANGenerator.state_dict(), 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state_dis, 'checkpoint/' + 'dCGANDiscriminator__' + str(epoch + 1))  # each epoch
        torch.save(state_gen, 'checkpoint/' + 'dCGANGenerator__' + str(epoch + 1))  # each epoch
