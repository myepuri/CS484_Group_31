# models/unsupervised/kmeans_model.py
import torch
from torchvision import models

def get_pretrained_resnet18(device='cpu'):
    
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
    model = model.to(device)
    model.eval()
    return model
