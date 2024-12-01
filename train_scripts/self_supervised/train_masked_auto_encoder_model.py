# train_scripts/self_supervised/train_masked_auto_encoder.py

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from models.utils.data_preprocessing import load_cifar10_data
import os
import time
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PCA_COMPONENTS,
    KMEANS_CLUSTERS,
    KMEANS_INIT,
    KMEANS_RANDOM_STATE,
    SELF_SUPERVISED_METRICS_DIR,
)
from models.utils.data_preprocessing import load_cifar10_data
from models.self_supervised.masked_auto_encoder_model import MaskedAutoEncoder
from models.utils.evaluation import evaluate_all_metrics


def evaluate_reconstruction_error(true_data, reconstructed_data):
    """Calculate Mean Squared Error (MSE) for reconstruction-based models."""
    true_data_flat = true_data.view(true_data.size(0), -1).detach().cpu().numpy()
    reconstructed_data_flat = reconstructed_data.view(reconstructed_data.size(0), -1).detach().cpu().numpy()
    return mean_squared_error(true_data_flat, reconstructed_data_flat)


def main():
    os.makedirs(SELF_SUPERVISED_METRICS_DIR, exist_ok=True)
    print(f"💻 Using device: {DEVICE}")

    # Load CIFAR-10 data
    print("📥 Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)

    # Initialize masked auto encoder model
    print("🔧 Initializing masked auto encoder model...")
    model = MaskedAutoEncoder(latent_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the masked auto encoder model
    print("🚀 Starting masked auto encoder model training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()

            # Compute masked prediction loss
            loss = model.reconstruction_loss(images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"✅ Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader):.4f}")
    print("✅ masked auto encoder model training completed.")

    # Evaluate the masked auto encoder model on the test dataset
    print("🔍 Evaluating the masked auto encoder model on test data...")
    features, true_labels = [], []
    reconstruction_errors = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            # Forward pass
            latent, reconstructed = model(images)
            latent_flat = latent.view(latent.size(0), -1)  # Flatten latent features
            features.append(latent_flat.cpu().numpy())
            true_labels.extend(labels.numpy())

            # Calculate reconstruction error (MSE) for the batch
            batch_reconstruction_error = evaluate_reconstruction_error(images, reconstructed)
            reconstruction_errors.append(batch_reconstruction_error)

    # Prepare evaluation inputs
    features = np.concatenate(features, axis=0).astype(np.float64)
    avg_reconstruction_error = np.mean(reconstruction_errors)

    # Dimensionality reduction with PCA
    print("📉 Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=PCA_COMPONENTS)
    features_reduced = pca.fit_transform(features)
    print("✅ PCA completed.")

    # Perform K-Means clustering on latent features
    print("🧪 Running K-Means clustering on latent features...")
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=KMEANS_RANDOM_STATE, init=KMEANS_INIT)
    start_time = time.time()
    cluster_labels = kmeans.fit_predict(features_reduced)
    end_time = time.time()
    print("✅ K-Means clustering completed.")

    # Evaluate and save metrics
    print("📊 Evaluating metrics and saving results...")
    evaluate_all_metrics(
        model_name="Masked Auto Encoder Model on CIFAR-10",
        start_time=start_time,
        end_time=end_time,
        output_file=f"{SELF_SUPERVISED_METRICS_DIR}/evaluation_results.txt",
        true_labels=true_labels,
        cluster_labels=cluster_labels,
        features=features_reduced,
        true_data=torch.cat([images.detach() for images, _ in test_loader], dim=0).to(DEVICE),
        reconstructed_data=torch.cat([model(images.to(DEVICE))[1].detach() for images, _ in test_loader], dim=0),
    )
    print(f"✅ Evaluation completed. Results saved in '{SELF_SUPERVISED_METRICS_DIR}/evaluation_results.txt'.")


if __name__ == "__main__":
    main()
