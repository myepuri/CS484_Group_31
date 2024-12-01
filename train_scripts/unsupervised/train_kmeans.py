# train_scripts/unsupervised/train_kmeans.py

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

from config import DEVICE, BATCH_SIZE, PCA_COMPONENTS, KMEANS_CLUSTERS, KMEANS_INIT, KMEANS_RANDOM_STATE, UNSUPERVISED_METRICS_DIR
from models.utils.data_preprocessing import load_cifar10_data
from models.unsupervised.kmeans_model import get_pretrained_resnet18
from models.utils.evaluation import evaluate_all_metrics

def extract_features(data_loader, model, device=DEVICE):
    
    features = []
    print("🔄 Extracting features using pre-trained ResNet-18...")
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            output = model(images)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            if i % 10 == 0:
                print(f"✅ Processed batch {i + 1}")
    print("✅ Feature extraction completed.")
    return np.concatenate(features)

def main():
    os.makedirs(UNSUPERVISED_METRICS_DIR, exist_ok=True)
    print(f"💻 Using device: {DEVICE}")

    # Load CIFAR-10 data
    print("📥 Loading CIFAR-10 data...")
    train_loader, _ = load_cifar10_data(batch_size=BATCH_SIZE)

    # Load pre-trained ResNet-18
    print("🔧 Loading pre-trained ResNet-18 model...")
    model = get_pretrained_resnet18(device=DEVICE)

    # Extract features
    features = extract_features(train_loader, model, device=DEVICE)

    # Dimensionality reduction with PCA
    print("📉 Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=PCA_COMPONENTS)
    features_reduced = pca.fit_transform(features)
    print("✅ PCA completed.")

    # K-Means clustering
    print("🧪 Running K-Means clustering...")
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, init=KMEANS_INIT, random_state=KMEANS_RANDOM_STATE)
    start_time = time.time()
    cluster_labels = kmeans.fit_predict(features_reduced)
    end_time = time.time()
    print("✅ K-Means clustering completed.")

    # True labels
    print("📝 Preparing true labels...")
    true_labels = [label for _, labels in train_loader for label in labels.numpy()]
    print(f"✅ Found {len(true_labels)} true labels.")

    # Evaluate and save metrics
    print("📊 Evaluating metrics and saving results...")
    evaluate_all_metrics(
        model_name="K-Means on Pre-Trained ResNet-18 Features",
        start_time=start_time,
        end_time=end_time,
        output_file=f"{UNSUPERVISED_METRICS_DIR}/evaluation_results.txt",
        true_labels=true_labels,
        cluster_labels=cluster_labels,
        features=features_reduced
    )
    print(f"✅ Evaluation completed. Results saved in '{UNSUPERVISED_METRICS_DIR}/evaluation_results.txt'.")

if __name__ == '__main__':
    main()
