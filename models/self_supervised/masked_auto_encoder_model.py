# models/self_supervised/masked_auto_encoder_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(MaskedAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downscale to 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downscale to 8x8
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64 channels, 8x8
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (batch_size, 64 * 8 * 8 = 4096)
            nn.Linear(4096, latent_dim)  # Project to latent_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),  # Map latent_dim back to 64 * 8 * 8
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),  # Reshape to (batch_size, 64, 8, 8)
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),  # Upscale to 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upscale to 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=1),  # Match input channels
            nn.Sigmoid()  # Output pixel values in range [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def reconstruction_loss(self, x, mask_ratio=0.25):
        
        batch_size, channels, height, width = x.size()
        mask = torch.rand(batch_size, height, width, device=x.device) < mask_ratio

        # Expand mask to match the channel dimension
        mask = mask.unsqueeze(1).expand(-1, channels, -1, -1)

        # Mask the input
        masked_x = x.clone()
        masked_x[mask] = 0  # Apply mask

        # Pass masked input through the model
        _, reconstructed = self.forward(masked_x)

        # Compute reconstruction loss only on masked regions
        loss = F.mse_loss(reconstructed[mask], x[mask])
        return loss
