import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the Data ---
# Loading 1.1 million rows can take a moment; use float32 to save memory
train_df = pd.read_csv('train.csv')

# --- 2. Structural Inspection ---
print("--- Dataset Info ---")
print(train_df.info())  # Check data types and memory usage
print("\n--- Summary Statistics ---")
print(train_df.describe())  # Check mean, min, max of the 21 factors

# --- 3. Cleanliness Check ---
print("\n--- Missing Values ---")
print(train_df.isnull().sum().sum()) # Should be 0 for this Kaggle dataset

# --- 4. Target Variable Analysis ---
# Important for XGBoost to see if the target is balanced
plt.figure(figsize=(10, 5))
sns.histplot(train_df['FloodProbability'], kde=True, color='blue')
plt.title("Distribution of Flood Probability (Target)")
plt.show()

# --- 5. Correlation Heatmap ---
# This helps you identify the "unnecessary factors" to drop later
plt.figure(figsize=(20, 12))
mask = np.triu(np.ones_like(train_df.corr(), dtype=bool)) # Hide the duplicate half
sns.heatmap(train_df.corr(), mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Map of All 21 Factors")
plt.show()

# --- 6. Relationship Ranking ---
correlations = train_df.corr()['FloodProbability'].sort_values(ascending=False)
print("\n--- Factor Correlation Ranking ---")
print(correlations)