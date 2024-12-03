import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import VISUALIZATION_DIR, UNSUPERVISED_METRICS_DIR, SELF_SUPERVISED_METRICS_DIR
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class EnhancedResultsVisualizer:
    def __init__(self, output_dir="results/visualizations"):
        #Initialize the visualizer with output directory.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define a professional color palette for improved aesthetics
        self.custom_palette = sns.color_palette("Set2")
        sns.set_palette(self.custom_palette)
        sns.set_context("notebook", font_scale=1.3)
        sns.set_style("whitegrid")

        # Shortened model names mapping
        self.model_name_mapping = {
            "K-Means on Pre-Trained ResNet-18 Features": "K-Means",
            "Autoencoder on CIFAR-10": "Autoencoder",
            "Contrastive Model on CIFAR-10": "SimCLR",
            "Masked Auto Encoder Model on CIFAR-10": "MAE",
        }

    def load_results(self, file_path):
        #Load evaluation results from a text file and format into a DataFrame.
        results = []
        current_model = None

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("Evaluation Results for"):
                    current_model = line.split("for ")[1].strip(":")
                elif current_model and line:
                    metric, value = line.split(": ")
                    results.append(
                        {
                            "Model": self.model_name_mapping.get(
                                current_model, current_model
                            ),
                            "Metric": metric,
                            "Value": float(value) if value != "N/A" else np.nan,
                        }
                    )
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Group by Model and Metric, compute the mean for duplicates
        aggregated_df = df.groupby(["Model", "Metric"], as_index=False).mean()
        return aggregated_df

    def plot_bar_charts(self, df):
        #Generate and save bar charts for each metric.
        metrics = df["Metric"].unique()

        for metric in metrics:
            metric_data = df[df["Metric"] == metric].dropna(subset=["Value"])
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(
                data=metric_data,
                x="Model",
                y="Value",
                palette=self.custom_palette,
                edgecolor="black",
            )

            # Display values on top of bars
            for bar in ax.patches:
                ax.annotate(
                    f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            plt.title(f"{metric} Comparison", fontsize=18, fontweight="bold")
            plt.xlabel("Model", fontsize=14)
            plt.ylabel(metric, fontsize=14)
            plt.xticks(rotation=30, ha="right", fontsize=12)
            plt.tight_layout()

            # Save the plot
            save_path = self.output_dir / f"{metric.replace(' ', '_').lower()}_bar_chart.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

    def plot_heatmap(self, df):
        #Generate and save a heatmap for all metrics.
        # Pivot data for heatmap
        pivot_data = df.pivot(index="Model", columns="Metric", values="Value")

        # Normalize data for a uniform color scale
        normalized_data = pivot_data.apply(
            lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0
        )

        # Plot the heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            normalized_data,
            annot=pivot_data.round(3),
            fmt=".3f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={"label": "Normalized Metric Score"},
            annot_kws={"fontsize": 10},
        )
        plt.title("Model Performance Heatmap", fontsize=20, fontweight="bold")
        plt.xlabel("Metrics", fontsize=14)
        plt.ylabel("Models", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Save the plot
        save_path = self.output_dir / "performance_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize(self, unsupervised_path, self_supervised_path):
        #Combine and visualize results from both unsupervised and self-supervised paths
        print("📊 Loading results...")
        unsupervised_df = self.load_results(unsupervised_path)
        self_supervised_df = self.load_results(self_supervised_path)

        # Combine results
        combined_df = pd.concat([unsupervised_df, self_supervised_df], ignore_index=True)

        print("📊 Generating bar charts...")
        self.plot_bar_charts(combined_df)

        print("📊 Generating heatmap...")
        self.plot_heatmap(combined_df)

        print("✅ Visualizations saved successfully in the output directory!")


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = EnhancedResultsVisualizer(output_dir="results/visualizations")

    # File paths for evaluation results
    unsupervised_path = "results/unsupervised_metrics/evaluation_results.txt"
    self_supervised_path = "results/self_supervised_metrics/evaluation_results.txt"

    # Generate visualizations
    visualizer.visualize(unsupervised_path, self_supervised_path)
