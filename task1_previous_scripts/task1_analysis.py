"""Task 1(c): descriptive statistics and exploratory plots for both datasets."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a window.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "task1_results"
RESULTS.mkdir(exist_ok=True)
DIABETES_OVERVIEW = RESULTS / "diabetes" / "overview"
DIGITS_OVERVIEW = RESULTS / "digits" / "overview"
DIABETES_OVERVIEW.mkdir(parents=True, exist_ok=True)
DIGITS_OVERVIEW.mkdir(parents=True, exist_ok=True)

diabetes = pd.read_csv(ROOT / "diabetes.csv")
digits = pd.read_csv(ROOT / "digits.csv")
diabetes_features = [column for column in diabetes if column != "target"]
digits_features = [column for column in digits if column != "target"]


def save_feature_stats(data, features, name):
    """Calculate mean, minimum, and maximum for every input feature."""
    stats = (
        data[features]
        .agg(["mean", "min", "max"])
        .T.rename_axis("feature")
        .reset_index()
    )
    stats_folder = RESULTS / name / "stats"
    stats_folder.mkdir(parents=True, exist_ok=True)
    path = stats_folder / f"{name}_feature_stats.csv"
    stats.to_csv(path, index=False)
    print(f"\n{name}: {len(data)} samples, {len(features)} features")
    print(f"Feature statistics saved to {path}")
    if name == "diabetes":
        print(f"Target: mean={data['target'].mean():.2f}, "
              f"min={data['target'].min():.2f}, max={data['target'].max():.2f}")
    else:
        print("Digit class counts:", data["target"].value_counts().sort_index().to_dict())
    return stats


def strongest_correlation(data, features):
    """Find the strongest feature pair, ignoring constant columns."""
    correlation = data[features].corr().abs()
    values = correlation.to_numpy().copy()
    upper = np.triu(np.ones(values.shape, dtype=bool), k=1)
    values[~upper] = np.nan
    row, col = np.unravel_index(np.nanargmax(values), values.shape)
    first, second = features[row], features[col]
    return first, second, data[first].corr(data[second])


diabetes_stats = save_feature_stats(diabetes, diabetes_features, "diabetes")
digits_stats = save_feature_stats(digits, digits_features, "digits")
print("\nDiabetes feature statistics:")
print(diabetes_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nDigits: all 64 feature statistics are in digits_feature_stats.csv")
print("Constant Digits pixels:",
      digits_stats.loc[digits_stats["min"] == digits_stats["max"], "feature"].tolist())
for name, data, features in (
    ("Diabetes", diabetes, diabetes_features),
    ("Digits", digits, digits_features),
):
    first, second, correlation = strongest_correlation(data, features)
    print(f"Strongest {name} feature correlation: "
          f"{first} and {second}, r={correlation:.3f}")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

# Diabetes: all ten input features are on the same scaled range.
fig, ax = plt.subplots(figsize=(10, 5))
diabetes[diabetes_features].boxplot(ax=ax, rot=45, grid=False)
ax.set(title="Diabetes: feature distributions (scaled values)",
       ylabel="Scaled feature value")
ax.axhline(0, color="gray", linewidth=0.8)
fig.tight_layout()
fig.savefig(DIABETES_OVERVIEW / "diabetes_boxplot.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, column in zip(axes.flat, ["age", "sex", "bmi", "bp", "s5", "target"]):
    ax.hist(diabetes[column], bins=20, color="#4472a8", edgecolor="white")
    ax.set(title=column, xlabel="Value", ylabel="Count")
fig.suptitle("Diabetes: selected feature and target histograms")
fig.tight_layout()
fig.savefig(DIABETES_OVERVIEW / "diabetes_histograms.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(diabetes["bmi"], diabetes["target"], alpha=0.5, s=20)
axes[0].set(xlabel="BMI (scaled)", ylabel="Disease progression",
            title="BMI versus target")
axes[1].scatter(diabetes["s1"], diabetes["s2"], alpha=0.5, s=20)
axes[1].set(xlabel="s1: total cholesterol (scaled)",
            ylabel="s2: LDL (scaled)", title="Two serum measurements")
fig.tight_layout()
fig.savefig(DIABETES_OVERVIEW / "diabetes_scatterplots.png", dpi=180)
plt.close(fig)

# Digits: select the pixels with the largest variance for readable plots.
selected_pixels = digits[digits_features].var().nlargest(6).index.tolist()
print("Digits pixels selected for plots:", selected_pixels)
fig, ax = plt.subplots(figsize=(9, 5))
digits[selected_pixels].boxplot(ax=ax, grid=False)
ax.set(title="Digits: six pixels with the largest variance",
       ylabel="Pixel intensity (0 to 16)")
fig.tight_layout()
fig.savefig(DIGITS_OVERVIEW / "digits_boxplot.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, column in zip(axes.flat, selected_pixels[:4]):
    ax.hist(digits[column], bins=np.arange(-0.5, 17.5, 1),
            color="#6b9b73", edgecolor="white")
    ax.set(title=column, xlabel="Pixel intensity", ylabel="Count", xlim=(-0.5, 16.5))
fig.suptitle("Digits: histograms of four variable pixels")
fig.tight_layout()
fig.savefig(DIGITS_OVERVIEW / "digits_histograms.png", dpi=180)
plt.close(fig)

pixels = digits[digits_features].to_numpy()
ink_count = (pixels > 0).sum(axis=1)
average_intensity = pixels.mean(axis=1)
fig, ax = plt.subplots(figsize=(8, 5.5))
points = ax.scatter(ink_count, average_intensity, c=digits["target"],
                    cmap="tab10", vmin=-0.5, vmax=9.5, alpha=0.55, s=18)
colorbar = fig.colorbar(points, ax=ax, ticks=range(10))
colorbar.set_label("Digit label")
ax.set(xlabel="Number of nonzero pixels", ylabel="Mean intensity across 64 pixels",
       title="Digits: image density by class")
fig.tight_layout()
fig.savefig(DIGITS_OVERVIEW / "digits_scatterplot.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
heatmap = ax.imshow(pixels.mean(axis=0).reshape(8, 8), cmap="magma")
fig.colorbar(heatmap, ax=ax, label="Mean pixel intensity")
ax.set(title="Digits: mean of each of the 64 pixels",
       xlabel="Image column", ylabel="Image row")
fig.tight_layout()
fig.savefig(DIGITS_OVERVIEW / "digits_mean_pixel_heatmap.png", dpi=180)
plt.close(fig)

print(f"\nAll tables and figures saved in {RESULTS}")
