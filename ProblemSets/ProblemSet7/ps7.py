# Full PS7 pipeline - improved version
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------- User-tunable settings ----------
DATA_PATH = "biden.csv"          # path to your dataset
RANDOM_STATE = 25
TEST_SIZE = 0.30

# Tuning complexity (match assignment)
N_ITER_DT = 100   # iterations for DecisionTree RandomizedSearchCV (assignment requires 100)
N_ITER_RF = 100   # iterations for RandomForest RandomizedSearchCV
CV = 5            # number of CV folds

# ---------- Load data ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find '{DATA_PATH}'. Download biden.csv and place in working directory.")

df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values and dtypes
print("\nMissing values per column:")
print(df.isna().sum())

print("\nColumn dtypes:")
print(df.dtypes)

# ---------- Prepare X, y (safe numeric-only + simple imputation) ----------
if 'biden' not in df.columns:
    raise KeyError("Expected target column named 'biden' in the CSV.")
y = df['biden']

# Keep only numeric predictors (safe for sklearn regressors)
X_all = df.drop(columns=['biden'])
numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in X_all.columns if c not in numeric_cols]
if non_numeric_cols:
    print(f"Warning: dropping non-numeric columns: {non_numeric_cols}")
X = X_all[numeric_cols].copy()

# Impute numeric NaNs with median (tree models handle this well)
if X.isna().any().any():
    print("Imputing numeric NaNs with column medians.")
    X = X.fillna(X.median())

# Drop any rows with NaN target
mask_target = y.notna()
if not mask_target.all():
    print(f"Dropping {(~mask_target).sum()} rows with NaN target.")
    X = X.loc[mask_target].reset_index(drop=True)
    y = y.loc[mask_target].reset_index(drop=True)
else:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

print(f"\nTarget variable (biden) stats:")
print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}, Min: {y.min()}, Max: {y.max()}")


# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTrain shape: {X_train.shape}; Test shape: {X_test.shape}")

# ---------- Part 1: simple decision tree ----------
print("\n" + "="*50)
print("PART 1: Basic Decision Tree")
print("="*50)

dt_simple = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE)
dt_simple.fit(X_train, y_train)
y_test_pred_simple = dt_simple.predict(X_test)
test_mse_simple = mean_squared_error(y_test, y_test_pred_simple)
print(f"Decision Tree (depth=3, min_samples_leaf=5) Test MSE: {test_mse_simple:.4f}")

# Plot the tree
plt.figure(figsize=(16, 8))
plot_tree(dt_simple, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=3, min_samples_leaf=5)")
plt.tight_layout()
plt.savefig("decision_tree_depth3.png", bbox_inches='tight', dpi=150)
plt.show()

# Print text version
print("\nText representation of tree:")
print(export_text(dt_simple, feature_names=list(X.columns)))

# ---------- Part 2: Tuned Decision Tree ----------
print("\n" + "="*50)
print("PART 2: Tuned Decision Tree (RandomizedSearchCV)")
print("="*50)

param_dist_dt = {
    'max_depth': [3, 10],  # Only these two values
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(2, 20)  # Starts at 2, not 1
}

dt_base = DecisionTreeRegressor(random_state=RANDOM_STATE)

rs_dt = RandomizedSearchCV(
    estimator=dt_base,
    param_distributions=param_dist_dt,
    n_iter=N_ITER_DT,
    scoring='neg_mean_squared_error',
    cv=CV,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
rs_dt.fit(X_train, y_train)
best_dt = rs_dt.best_estimator_
best_dt_params = rs_dt.best_params_
cv_mse_best_dt = -rs_dt.best_score_
test_mse_best_dt = mean_squared_error(y_test, best_dt.predict(X_test))


print("\nBest Decision Tree parameters:", best_dt_params)
print(f"Best Decision Tree CV MSE: {cv_mse_best_dt:.4f}")
print(f"Best Decision Tree Test MSE: {test_mse_best_dt:.4f}")

# Save and plot best tree
joblib.dump(best_dt, "best_decision_tree.pkl")

# Only plot if tree depth is reasonable
if best_dt.get_depth() <= 10:
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, feature_names=X.columns, filled=True, rounded=True, fontsize=8)
    plt.title(f"Best Decision Tree (depth={best_dt.get_depth()})")
    plt.tight_layout()
    plt.savefig("best_decision_tree.png", bbox_inches='tight', dpi=150)
    plt.show()
else:
    print(f"Tree too deep ({best_dt.get_depth()}) to plot clearly. Check feature importances instead.")

# ---------- Part 3: Tuned Random Forest ----------
print("\n" + "="*50)
print("PART 3: Tuned Random Forest (RandomizedSearchCV)")
print("="*50)

param_dist_rf = {
    'n_estimators': [10, 200],  # Only these two values
    'max_depth': [3, 10],       # Only these two values  
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(2, 20),
    'max_features': sp_randint(1, 5)  # Integers 1-4 only
}

rf_base = RandomForestRegressor(random_state=RANDOM_STATE)

rs_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist_rf,
    n_iter=N_ITER_RF,
    scoring='neg_mean_squared_error',
    cv=CV,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

rs_rf.fit(X_train, y_train)
best_rf = rs_rf.best_estimator_
best_rf_params = rs_rf.best_params_
cv_mse_best_rf = -rs_rf.best_score_
test_mse_best_rf = mean_squared_error(y_test, best_rf.predict(X_test))


print("\nBest Random Forest parameters:", best_rf_params)
print(f"Best Random Forest CV MSE: {cv_mse_best_rf:.4f}")
print(f"Best Random Forest Test MSE: {test_mse_best_rf:.4f}")

# Save RF model
joblib.dump(best_rf, "best_random_forest.pkl")

# Plot feature importances
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig("rf_feature_importances.png", bbox_inches='tight', dpi=150)
plt.show()

print("\nFeature Importances (Random Forest):")
for _, row in feature_importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ---------- Final Summary ----------
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

results = {
    'Model': ['Simple Decision Tree', 'Tuned Decision Tree', 'Random Forest'],
    'Test MSE': [test_mse_simple, test_mse_best_dt, test_mse_best_rf],
    'CV MSE': ['-', cv_mse_best_dt, cv_mse_best_rf]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

def pct_improvement(base, new):
    if base == 0:
        return float('nan')
    return ( (base - new) / base ) * 100.0

improvement_dt = pct_improvement(test_mse_simple, test_mse_best_dt)
improvement_rf_vs_simple = pct_improvement(test_mse_simple, test_mse_best_rf)
improvement_rf_vs_dt = pct_improvement(test_mse_best_dt, test_mse_best_rf)

print(f"Tuned DT vs Simple DT: {improvement_dt:+.1f}%")
print(f"Random Forest vs Simple DT: {improvement_rf_vs_simple:+.1f}%")
print(f"Random Forest vs Tuned DT: {improvement_rf_vs_dt:+.1f}%")


# Check saved files
print("\nSaved files:")
saved_candidates = ["decision_tree_depth3.png", "rf_feature_importances.png",
                    "best_decision_tree.pkl", "best_random_forest.pkl"]
# include best_decision_tree.png only if it was created earlier
if 'best_dt' in locals() and best_dt.get_depth() <= 10:
    saved_candidates.insert(1, "best_decision_tree.png")

for fname in saved_candidates:
    exists = "✓" if os.path.exists(fname) else "✗"
    print(f"  {exists} {fname}")

print("\nAnalysis complete!")
