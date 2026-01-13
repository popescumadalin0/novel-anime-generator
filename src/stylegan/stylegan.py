from stylegan.hyperparameters import NUM_EPOCHS, BATCH_SIZE, LATENT_DIM
from stylegan.model import train_stylegan

styleGANGenerator, styleGANDiscriminator = train_stylegan(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM
    )