import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.calibration import calibration_curve

# Verify file exists
filename = 'Mapping_Injury__Overdose__and_Violence_-_Census_Tract.csv'
if not os.path.exists(filename):
    raise FileNotFoundError(f"File not found: {filename}. Please ensure it is uploaded and the path is correct.")

# Load dataset
df = pd.read_csv(filename)

# Filter Virginia and suicide intent
va = df[(df.ST_NAME == 'Virginia') & (df.Intent == 'All_Suicide')]

# Pivot to have one row per tract, columns: Rate_2022, Rate_2023
pivot = va.pivot(index='GEOID', columns='Period', values='Rate').rename(columns={'2022':'Rate_2022','2023':'Rate_2023'})

# Drop any tracts missing values
pivot = pivot.dropna()

# Create binary label: top 10% rate in 2023
threshold = np.percentile(pivot['Rate_2023'], 90)
pivot['label'] = (pivot['Rate_2023'] >= threshold).astype(int)

# Features: use Rate_2022 as a simple baseline
X = pivot[['Rate_2022']].values
y = pivot['label'].values

# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, pivot.index, test_size=0.2, stratify=y, random_state=42
)

# Baseline model: logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

# Metrics
auroc = roc_auc_score(y_test, y_proba)
precision_10 = precision_score(y_test, y_pred)
recall_10 = recall_score(y_test, y_pred)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

# Error analysis: false positives and false negatives
errors = pd.DataFrame({
    'GEOID': idx_test,
    'Rate_2022': X_test.flatten(),
    'Rate_2023': pivot.loc[idx_test, 'Rate_2023'],
    'y_true': y_test,
    'y_pred': y_pred,
    'y_proba': y_proba
}).set_index('GEOID')

false_positives = errors[(errors.y_true == 0) & (errors.y_pred == 1)].sort_values('y_proba', ascending=False)
false_negatives = errors[(errors.y_true == 1) & (errors.y_pred == 0)].sort_values('y_proba')

# Print summary
print(f"AUROC: {auroc:.3f}")
print(f"Precision (threshold=0.5): {precision_10:.3f}")
print(f"Recall (threshold=0.5): {recall_10:.3f}")
print("\nCalibration curve (prob_pred vs prob_true):")
for bp, bt in zip(prob_pred, prob_true):
    print(f" {bp:.2f} -> {bt:.2f}")

print("\nTop 5 False Positives:")
print(false_positives.head())
print("\nTop 5 False Negatives:")
print(false_negatives.head())
