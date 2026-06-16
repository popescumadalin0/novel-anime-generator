# Novel Anime Generator

A deep learning project for generating anime-style face images using multiple GAN architectures: **DCGAN**, **Conditional DCGAN**, and **StyleGAN**.

## Overview

This project explores generative adversarial networks applied to anime face generation. It implements and compares three different GAN architectures, including conditional generation that allows controlling attributes such as hair color and eye color.

## Project Structure

```
src/
  dcgan/
      DCGAN_generator.py       Generator architecture
          DCGAN_discriminator.py   Discriminator architecture
              dcgan.py                 Training loop
                  model.py                 Model definitions
                      data_loader.py           Dataset loading and preprocessing
                          hyperparameters.py       Training hyperparameters
                              optimizers.py            Optimizer configuration
                                  utils.py                 Utility functions
                                      AnimeData.py             Anime dataset handler
                                        stylegan/                  StyleGAN implementation
                                          conditional_anime_gan.py   Conditional DCGAN (attribute-conditioned)
                                            latent_space_interpolation.py  Latent space exploration
                                            report/                      Project report and results
                                            requirements.txt             Python dependencies
                                            ```

                                            ## Models

                                            - **DCGAN** — Standard Deep Convolutional GAN for unconditional anime face generation
                                            - **Conditional DCGAN** — Class-conditioned generation allowing control over attributes (e.g. hair color, eye color) via label embeddings
                                            - **StyleGAN** — Style-based GAN for higher quality and more controllable generation
                                            - **Latent Space Interpolation** — Tool for exploring smooth transitions between generated images

                                            ## Technologies

                                            - Python
                                            - PyTorch
                                            - torchvision
                                            - Kaggle Anime Face Dataset

                                            ## Setup

                                            1. Clone the repository:
                                               ```bash
                                                  git clone https://github.com/popescumadalin0/novel-anime-generator.git
                                                     cd novel-anime-generator
                                                        ```

                                                        2. Install dependencies:
                                                           ```bash
                                                              pip install -r requirements.txt
                                                                 ```

                                                                 3. Configure Kaggle API credentials in `src/kaggle.json` to download the dataset automatically.

                                                                 4. Run training:
                                                                    ```bash
                                                                       python src/dcgan/dcgan.py
                                                                          ```
