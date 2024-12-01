# models/utils/evaluation.py
import time
import torch
import psutil
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import numpy as np


def log_resource_usage():
    #Return GPU and CPU memory usage
    gpu_memory = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1e6  # Memory in MB
    return gpu_memory, cpu_memory


def evaluate_clustering(true_labels, cluster_labels):
    #Calculate the Adjusted Rand Index (ARI) between the true and predicted labels
    return adjusted_rand_score(true_labels, cluster_labels)


def evaluate_silhouette(features, cluster_labels):
    #Calculate the Silhouette Score for the clustering results
    return silhouette_score(features, cluster_labels)


def evaluate_nmi(true_labels, cluster_labels):
    #Calculate the Normalized Mutual Information (NMI) between the true and predicted labels
    return normalized_mutual_info_score(true_labels, cluster_labels)


def evaluate_clustering_accuracy(true_labels, cluster_labels, n_clusters=10):
    #Calculate clustering accuracy using the Hungarian algorithm for optimal label mapping
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for true, pred in zip(true_labels, cluster_labels):
        contingency_matrix[true][pred] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy


def evaluate_reconstruction_error(true_data, reconstructed_data):
    #Calculate Mean Squared Error (MSE) for reconstruction-based models.
    true_data_flat = true_data.view(true_data.size(0), -1).cpu().numpy()
    reconstructed_data_flat = reconstructed_data.view(reconstructed_data.size(0), -1).cpu().numpy()
    return mean_squared_error(true_data_flat, reconstructed_data_flat)


def evaluate_linear_classification_accuracy(latent_features, labels):
    #Train a logistic regression model on latent features and evaluate its accuracy
    latent_features = latent_features.cpu().numpy()
    labels = labels.cpu().numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(latent_features, labels)
    predictions = clf.predict(latent_features)
    return accuracy_score(labels, predictions)


def evaluate_execution_time(start_time, end_time):
    #Calculate execution time in seconds.
    return end_time - start_time


def save_evaluation_results(model_name, results, filename):
    #Save evaluation metrics to a specified text file
    with open(filename, 'a') as f:
        f.write(f"Evaluation Results for {model_name}:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
    print(f"Evaluation results saved to {filename}")


def evaluate_all_metrics(model_name, start_time, end_time, output_file, **kwargs):
    
    results = {}
    
    # Clustering metrics
    if "true_labels" in kwargs and "cluster_labels" in kwargs:
        results["Adjusted Rand Index (ARI)"] = evaluate_clustering(kwargs["true_labels"], kwargs["cluster_labels"])
        results["Silhouette Score"] = evaluate_silhouette(kwargs["features"], kwargs["cluster_labels"])
        results["NMI"] = evaluate_nmi(kwargs["true_labels"], kwargs["cluster_labels"])
        results["Clustering Accuracy"] = evaluate_clustering_accuracy(kwargs["true_labels"], kwargs["cluster_labels"])

    # Reconstruction error
    if "true_data" in kwargs and "reconstructed_data" in kwargs:
        results["Reconstruction Error (MSE)"] = evaluate_reconstruction_error(kwargs["true_data"], kwargs["reconstructed_data"])

    # Linear classification accuracy
    if "latent_features" in kwargs and "labels" in kwargs:
        results["Linear Classification Accuracy"] = evaluate_linear_classification_accuracy(kwargs["latent_features"], kwargs["labels"])

    # Execution time and resource usage
    results["Execution Time (s)"] = evaluate_execution_time(start_time, end_time)
    gpu_memory, cpu_memory = log_resource_usage()
    results["GPU Memory Used (MB)"] = gpu_memory
    results["CPU Memory Used (MB)"] = cpu_memory

    # Save results
    save_evaluation_results(model_name, results, output_file)
    return results
