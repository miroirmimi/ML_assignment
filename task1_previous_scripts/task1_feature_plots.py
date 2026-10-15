"""Save three plots for every Diabetes and Digits input feature."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "task1_results"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_diabetes_feature(data, feature, folders):
    values = data[feature]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.boxplot(values, tick_labels=[feature], showmeans=True)
    ax.set(title=f"Diabetes: {feature} boxplot", ylabel="Scaled feature value")
    save_plot(fig, folders["boxplots"] / f"{feature}.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = 2 if feature == "sex" else 25
    ax.hist(values, bins=bins, color="#4472a8", edgecolor="white")
    ax.set(title=f"Diabetes: {feature} histogram",
           xlabel="Scaled feature value", ylabel="Number of patients")
    save_plot(fig, folders["histograms"] / f"{feature}.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(values, data["target"], s=20, alpha=0.55, color="#4472a8")
    ax.set(title=f"Diabetes: {feature} versus target",
           xlabel=f"{feature} (scaled)", ylabel="Disease progression after one year")
    save_plot(fig, folders["scatterplots"] / f"{feature}.png")


def plot_digits_feature(data, feature, folders, class_jitter):
    values = data[feature]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.boxplot(values, tick_labels=[feature], showmeans=True)
    ax.set(title=f"Digits: {feature} boxplot", ylabel="Pixel intensity (0 to 16)",
           ylim=(-0.5, 16.5))
    save_plot(fig, folders["boxplots"] / f"{feature}.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(values, bins=np.arange(-0.5, 17.5, 1),
            color="#6b9b73", edgecolor="white")
    ax.set(title=f"Digits: {feature} histogram", xlabel="Pixel intensity",
           ylabel="Number of images", xlim=(-0.5, 16.5))
    save_plot(fig, folders["histograms"] / f"{feature}.png")

    # Small horizontal jitter separates overlapping class labels visually.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["target"] + class_jitter, values,
               s=8, alpha=0.18, color="#547caa")
    ax.set(title=f"Digits: {feature} by digit class",
           xlabel="Digit class (jittered for visibility)",
           ylabel="Pixel intensity", xticks=range(10),
           xlim=(-0.5, 9.5), ylim=(-0.5, 16.5))
    save_plot(fig, folders["scatterplots"] / f"{feature}.png")


def main():
    rng = np.random.default_rng(42)
    for dataset_name in ("diabetes", "digits"):
        data = pd.read_csv(ROOT / f"{dataset_name}.csv")
        features = [column for column in data if column != "target"]
        folders = {
            chart_type: OUTPUT / dataset_name / chart_type
            for chart_type in ("boxplots", "histograms", "scatterplots")
        }
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        if dataset_name == "diabetes":
            for feature in features:
                plot_diabetes_feature(data, feature, folders)
        else:
            class_jitter = rng.uniform(-0.18, 0.18, size=len(data))
            for feature in features:
                plot_digits_feature(data, feature, folders, class_jitter)

        print(f"{dataset_name}: {len(features)} features × 3 plot types "
              f"= {len(features) * 3} figures")
        for chart_type, folder in folders.items():
            print(f"  {chart_type}: {folder}")


if __name__ == "__main__":
    main()
