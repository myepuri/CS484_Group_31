# models/unsupervised/autoencoder_model.py
import torch
import torch.nn as nn
from config import DEVICE

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Downscale to 16x16
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downscale to 8x8
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downscale to 4x4
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64)  # Output latent_dim = 64
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 4 * 4),  # Match the encoder's output
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscale to 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscale to 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscale to 32x32
            nn.Sigmoid()
        )


    def forward(self, x):
        
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
