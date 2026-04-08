import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. RELOAD THE DATA (Fast loading logic with reduced memory)
# We exclude 'id', 'CoastalVulnerability', and 'Landslides' immediately
cols_to_drop = ['id', 'CoastalVulnerability', 'Landslides']
full_cols = pd.read_csv('train.csv', nrows=0).columns
use_cols = [c for c in full_cols if c not in cols_to_drop]

print("Loading 1.1M rows efficiently...")
df = pd.read_csv('train.csv', usecols=use_cols, dtype=np.float32)

# Separate features and target
X = df.drop(columns=['FloodProbability'])
y = df['FloodProbability'].values

# 2. ADVANCED FEATURE ENGINEERING (Logic for 0.84+ R2 Score)
# These features help the model understand the interactions between environmental factors
print("Applying Advanced Feature Engineering...")
X['fsum'] = X.values.sum(axis=1)                      # Total environmental pressure
X['std'] = X.drop(columns=['fsum']).std(axis=1)       # Detects extreme spikes in individual factors
X['fsum_sq'] = X['fsum'] ** 2                         # Emphasizes high-risk combinations

# 3. SCALE
# Essential to keep the engineered features (like fsum_sq) in the same range as others
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. SPLIT (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. INITIALIZE TUNED XGBOOST
# Optimized parameters for large-scale regression
print("Initializing Tuned XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=1000,      # More trees for better learning
    learning_rate=0.03,     # Slower, more precise steps
    max_depth=6,            # Depth 6 is stable with engineered features
    subsample=0.8,          # Prevents overfitting by using 80% of data per tree
    colsample_bytree=0.8,   # Uses 80% of features per tree
    tree_method='hist',     # High-performance mode for 1.1M rows
    random_state=42
)

# 6. TRAIN THE MODEL
print("Training started on 1.1 Million rows... (This may take 2-4 minutes)")
model.fit(X_train, y_train)

# 7. EVALUATE
y_pred = model.predict(X_val)
print("\n--- FINAL MODEL PERFORMANCE ---")
print(f"R2 Score: {r2_score(y_val, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_val, y_pred):.6f}")

# 8. SAVE THE MODEL
# Saving as .json allows you to load this "brain" into your Streamlit app later
model.save_model("flood_model.json")
print("\nModel saved as: flood_model.json")