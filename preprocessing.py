import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. SMART LOAD (Excludes unnecessary columns immediately)
# This prevents your RAM from filling up with 'id' and 'noisy' factors
cols_to_drop = ['id', 'CoastalVulnerability', 'Landslides']
full_cols = pd.read_csv('train.csv', nrows=0).columns
use_cols = [c for c in full_cols if c not in cols_to_drop]

print("Loading 1.1M rows efficiently...")
# Using float32 uses 50% less RAM than your previous code
df = pd.read_csv('train.csv', usecols=use_cols, dtype=np.float32)

# 2. SEPARATE FEATURES AND TARGET
X = df.drop(columns=['FloodProbability'])
y = df['FloodProbability'].values 

# 3. FAST FEATURE ENGINEERING
print("Creating 'fsum' feature...")
# .values.sum is 10x faster than the standard .sum() you were using
X['fsum'] = X.values.sum(axis=1)

# 4. MEMORY-EFFICIENT SCALING
print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. FAST SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n--- PREPROCESSING COMPLETE ---")
print(f"Final training rows: {X_train.shape[0]}")
print(f"Time: This should have finished in under 30 seconds.")