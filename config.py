
# General Configuration
import torch

try:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Warning: Issue with device selection. Defaulting to CPU. Error: {e}")
    DEVICE = torch.device('cpu')

# Data Configuration
DATA_DIR = './data/cifar10'  # Directory to store CIFAR-10 dataset
BATCH_SIZE = 32  # Batch size for training and evaluation

# Model Training Configuration
LEARNING_RATE = 0.001  # Learning rate for all models
NUM_EPOCHS = 10  # Number of epochs for training all models
WEIGHT_DECAY = 1e-4  # Weight decay for optimizers (if applicable)

# PCA Configuration
PCA_COMPONENTS = 50  # Number of components for PCA dimensionality reduction

# Clustering Configuration
KMEANS_CLUSTERS = 10  # Number of clusters for K-Means
KMEANS_INIT = 'k-means++'  # Initialization method for K-Means
KMEANS_RANDOM_STATE = 0  # Random state for K-Means reproducibility

# Logging and Results
RESULTS_DIR = './results'  # Directory to store results
UNSUPERVISED_METRICS_DIR = f'{RESULTS_DIR}/unsupervised_metrics'
SELF_SUPERVISED_METRICS_DIR = f'{RESULTS_DIR}/self_supervised_metrics'
VISUALIZATION_DIR = f'{RESULTS_DIR}/visualizations'

# Training ResNet-18
RESNET_NUM_CLASSES = 10  # Number of output classes for ResNet-18 when training on CIFAR-10

SEED = 42  # Seed for reproducibility across the project
