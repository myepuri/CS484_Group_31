# train_scripts/unsupervised/train_autoencoder.py

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from models.utils.data_preprocessing import load_cifar10_data
import os
import time
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PCA_COMPONENTS, KMEANS_CLUSTERS, KMEANS_INIT, KMEANS_RANDOM_STATE, UNSUPERVISED_METRICS_DIR
from models.utils.data_preprocessing import load_cifar10_data
from models.unsupervised.autoencoder_model import Autoencoder
from models.utils.evaluation import evaluate_all_metrics

def main():
    os.makedirs(UNSUPERVISED_METRICS_DIR, exist_ok=True)
    print(f"💻 Using device: {DEVICE}")

    # Load CIFAR-10 data
    print("📥 Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)

    # Initialize Autoencoder model
    print("🔧 Initializing Autoencoder model...")
    model = Autoencoder(latent_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Train the Autoencoder
    print("🚀 Starting Autoencoder training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            _, reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"✅ Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader):.4f}")
    print("✅ Autoencoder training completed.")

    # Evaluate the Autoencoder on the test dataset
    print("🔍 Evaluating the Autoencoder on test data...")
    features, true_labels, original_images, reconstructed_images = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            latent, reconstructed = model(images)
            features.append(latent.cpu().numpy())
            true_labels.extend(labels.numpy())
            original_images.append(images.cpu())
            reconstructed_images.append(reconstructed.cpu())

    # Prepare evaluation inputs
    features = np.concatenate(features, axis=0).astype(np.float64)
    original_images = torch.cat(original_images)
    reconstructed_images = torch.cat(reconstructed_images)

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
        model_name="Autoencoder on CIFAR-10",
        start_time=start_time,
        end_time=end_time,
        output_file=f"{UNSUPERVISED_METRICS_DIR}/evaluation_results.txt",
        true_labels=true_labels,
        cluster_labels=cluster_labels,
        features=features_reduced,
        true_data=original_images,
        reconstructed_data=reconstructed_images,
        latent_features=torch.tensor(features),
        labels=torch.tensor(true_labels)
    )
    print(f"✅ Evaluation completed. Results saved in '{UNSUPERVISED_METRICS_DIR}/evaluation_results.txt'.")

if __name__ == '__main__':
    main()
