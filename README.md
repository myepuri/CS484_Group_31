# Unsupervised and Self-Supervised Learning on CIFAR-10

A comprehensive implementation of unsupervised and self-supervised learning approaches for feature learning and clustering using the CIFAR-10 dataset.

## Features

- Implementation of multiple learning approaches:
  - K-means clustering with pretrained ResNet features
  - Autoencoder for unsupervised feature learning 
  - SimCLR (Simple Contrastive Learning of Representations)
  - MAE (Masked Autoencoder)
- Comprehensive evaluation metrics and visualization tools
- Modular architecture for easy extension

## Project Structure

```
ML_Project/
├── data/
│   └── cifar10/          # CIFAR-10 dataset storage
├── models/
│   ├── unsupervised/     # Traditional unsupervised models
│   ├── self_supervised/  # Self-supervised implementations
│   └── utils/           # Helper functions and utilities
├── results/
│   ├── unsupervised_metrics/
│   ├── self_supervised_metrics/
│   └── visualizations/
└── train_scripts/        # Training implementations
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv

   Set Execution Policy: Run the following command to temporarily allow running scripts:
   powershell
   cmd: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
      This will only affect the current PowerShell session.
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data/cifar10
   mkdir -p results/unsupervised_metrics
   mkdir -p results/self_supervised_metrics
   mkdir -p results/visualizations
   ```

## Running the Models

### 1. K-means Clustering
```bash
python train_scripts/unsupervised/train_kmeans.py
```

### 2. Autoencoder
```bash
python train_scripts/unsupervised/train_autoencoder.py
```

### 3. SimCLR (Contrastive Learning)
```bash
python train_scripts/self_supervised/train_contrastive_model.py
```

### 4. Masked Autoencoder (MAE)
```bash
python train_scripts/self_supervised/train_masked_prediction_model.py
```

### 5. Visualize Results
```bash
python visualize_results.py
```

## Configuration

Modify `config.py` to adjust:
- Batch size
- Learning rate
- Number of epochs
- PCA components
- Number of clusters
- Device settings (CPU/GPU)

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## Model Architectures

### SimCLR
- Encoder: CNN with multiple convolutional layers
- Projection head: MLP with ReLU activation
- NT-Xent loss for contrastive learning

### MAE
- Asymmetric encoder-decoder architecture
- Random masking of input patches
- Reconstruction-based learning objective

### Autoencoder
- Symmetric encoder-decoder architecture
- Convolutional layers for feature extraction
- MSE loss for reconstruction

## Results

Results are saved in the following locations:
- Metrics: `results/[unsupervised|self_supervised]_metrics/`
- Visualizations: `results/visualizations/`
- Model checkpoints: `results/models/`