import pandas as pd
import numpy as np

# 1. Load the CSV file
data = pd.read_csv("data/store_sales.csv")


# 2. Preview the data
print("First 5 rows of the dataset:")
print(data.head())

# 3. Extract features and target
# Features: TotalGasGallons, LottoSales, DayType
X = data[['TotalGasGallons', 'LottoSales', 'DayType']].values

# Target: TotalSales
y = data['TotalSales'].values

# 4. Add intercept term (column of 1s)
X = np.c_[np.ones(X.shape[0]), X]

# 5. Check shapes
print(f"\nFeatures shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# 6. Optional: print first row
print("\nFirst row of features (with intercept):", X[0])
print("First target value:", y[0])
