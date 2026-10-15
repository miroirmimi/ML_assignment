from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, load_digits

diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
df_diabetes = pd.DataFrame(X_diabetes, columns=diabetes.feature_names)
df_diabetes["target"] = y_diabetes
print(f"Diabetes features: {list(diabetes.feature_names)}")

# Check feature values only; remove a row if any feature is missing.
diabetes_features = list(diabetes.feature_names)
diabetes_missing = df_diabetes[diabetes_features].isna().any(axis=1)
print(f"Diabetes rows with missing features: {diabetes_missing.sum()} / {len(df_diabetes)}")
df_diabetes = df_diabetes.dropna(subset=diabetes_features).copy()
X_diabetes = df_diabetes[diabetes_features].to_numpy()
y_diabetes = df_diabetes["target"].to_numpy()
print(f"Diabetes rows after cleaning: {len(df_diabetes)}")

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
print(f"Digits shape: {X_digits.shape}")

digits_features = [f"pixel_{i}" for i in range(X_digits.shape[1])]
df_digits = pd.DataFrame(X_digits, columns=digits_features)
df_digits["target"] = y_digits
digits_missing = df_digits[digits_features].isna().any(axis=1)
print(f"Digits rows with missing features: {digits_missing.sum()} / {len(df_digits)}")
df_digits = df_digits.dropna(subset=digits_features).copy()
X_digits = df_digits[digits_features].to_numpy()
y_digits = df_digits["target"].to_numpy()
print(f"Digits rows after cleaning: {len(df_digits)}")

# Save one complete dataset per CSV in the same folder as this script.
output_dir = Path(__file__).resolve().parent
df_diabetes.to_csv(output_dir / "diabetes.csv", index=False)
df_digits.to_csv(output_dir / "digits.csv", index=False)
print(f"Saved: {output_dir / 'diabetes.csv'}")
print(f"Saved: {output_dir / 'digits.csv'}")

diabetes_raw = load_diabetes(scaled=False)
df_raw = pd.DataFrame(diabetes_raw.data, columns=diabetes_raw.feature_names)
df_raw.to_csv(output_dir / "diabetes_raw.csv", index=False)
