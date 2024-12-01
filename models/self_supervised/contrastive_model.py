# models/self_supervised/contrastive_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE

class ContrastiveModel(nn.Module):
    def __init__(self, latent_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downscale to 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downscale to 8x8
        )
        self.projector = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Intermediate layer
            nn.ReLU(),
            nn.Linear(128, 64)  # Output latent_dim = 64
        )


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)

    def contrastive_loss(self, features, temperature=0.5):
    
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        similarity_matrix_exp = torch.exp(similarity_matrix)

        # Mask to ignore self-similarity
        mask = ~torch.eye(batch_size, device=features.device).bool()

        # Numerator: Positive pair similarities
        numerator = similarity_matrix_exp[mask].view(batch_size, -1)

        # Denominator: All pair similarities
        denominator = similarity_matrix_exp.sum(dim=1, keepdim=True).expand_as(numerator)

        # Contrastive loss
        loss = -torch.log(numerator / denominator).mean()
        return loss
