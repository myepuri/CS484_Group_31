# train_scripts/self_supervised/train_constrastive_model.py

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

from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PCA_COMPONENTS, KMEANS_CLUSTERS, KMEANS_INIT, KMEANS_RANDOM_STATE, SELF_SUPERVISED_METRICS_DIR
from models.utils.data_preprocessing import load_cifar10_data
from models.self_supervised.contrastive_model import ContrastiveModel
from models.utils.evaluation import evaluate_all_metrics

def main():
    os.makedirs(SELF_SUPERVISED_METRICS_DIR, exist_ok=True)
    print(f"💻 Using device: {DEVICE}")

    # Load CIFAR-10 data
    print("📥 Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10_data(batch_size=BATCH_SIZE)

    # Initialize Contrastive Model
    print("🔧 Initializing Contrastive Model...")
    model = ContrastiveModel(latent_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the Contrastive Model
    print("🚀 Starting Contrastive Model training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()

            # Encode features and compute contrastive loss
            features = model.encode(images)
            loss = model.contrastive_loss(features)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"✅ Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader):.4f}")
    print("✅ Contrastive Model training completed.")

    # Evaluate the Contrastive Model on the test dataset
    print("🔍 Evaluating the Contrastive Model on test data...")
    features, true_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            latent = model.encode(images)
            features.append(latent.cpu().numpy())
            true_labels.extend(labels.numpy())

    # Prepare evaluation inputs
    features = np.concatenate(features, axis=0).astype(np.float64)
    true_labels = np.array(true_labels)

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
        model_name="Contrastive Model on CIFAR-10",
        start_time=start_time,
        end_time=end_time,
        output_file=f"{SELF_SUPERVISED_METRICS_DIR}/evaluation_results.txt",
        true_labels=true_labels,
        cluster_labels=cluster_labels,
        features=features_reduced,
        latent_features=torch.tensor(features),
        labels=torch.tensor(true_labels)
    )
    print(f"✅ Evaluation completed. Results saved in '{SELF_SUPERVISED_METRICS_DIR}/evaluation_results.txt'.")

if __name__ == '__main__':
    main()
