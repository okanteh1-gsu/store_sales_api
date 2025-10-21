import numpy as np
from .readcvs import X, y

def gradient_descent_fit(X, y, alpha=0.001, tolerance=1e-6):
    X_new = X[:, 1:]
    X_mean = np.mean(X_new, axis=0)
    X_std = np.std(X_new, axis=0)
    X_standardized = (X_new - X_mean) / X_std

    # Add intercept
    X_standardized = np.c_[np.ones(X_standardized.shape[0]), X_standardized]

    n_rows, n_cols = X_standardized.shape
    w = np.zeros(n_cols)
    change = float('inf')
    iter_count = 0

    while change > tolerance:

        y_pred = X_standardized @ w
        error = y - y_pred
        gradient = -(2 / n_rows) * (X_standardized.T @ error)

        w_new = w - alpha * gradient

        change = np.max(np.abs(w_new - w))
        w = w_new
        iter_count += 1



    print(f"Took {iter_count} iterations to converge")
    print("Weights (standardized):", w)

    B_unstandardized = w.copy()
    B_unstandardized[1:] = w[1:] / X_std
    B_unstandardized[0] = w[0] - np.sum((w[1:] * X_mean) / X_std)
    print("weight: ", B_unstandardized)

    return B_unstandardized




# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8,5))
# plt.plot(y, 'o-', label='Actual Sales')
# plt.plot(y_pred_all, 's--', label='Predicted Sales')
# plt.xlabel('Day Index')
# plt.ylabel('Total Sales')
# plt.title('Actual vs Predicted Sales')
# plt.legend()
# plt.grid(True)
# plt.show()
