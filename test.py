# PyTorch documentation
import torch
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)
# print(np.__version__)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
# without un-squeeze, errors will happen later on (shapes within linear layers)
# X = torch.arange(start, end, step).unsqueeze(dim=1)
X = torch.from_numpy(np.genfromtxt("results_L1.csv", delimiter=",", dtype="int8"))
print(X[:5])
# y = weight * X + bias
# print(f"{X[:5]}, {y[:5]}")
#
# # Prepare train data and test data
# train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
# print(f"{train_split}")
# X_train, y_train = X[:train_split], y[:train_split]
# X_test, y_test = X[train_split:], y[train_split:]
# print(f"{len(X_train)}, {len(y_train)}, {len(X_test)}, {len(y_test)}")
#
#
# def plot_predictions(train_data=X_train,
#                      train_labels=y_train,
#                      test_data=X_test,
#                      test_labels=y_test,
#                      predictions=None):
#
#     """
#     Plots training data, test data and compares predictions.
#     """
#     plt.figure(figsize=(8, 6))
#
#     # Plot training data in blue
#     plt.scatter(train_data, train_labels, c="b", s=6, label="Training data")
#
#     # Plot test data in green
#     plt.scatter(test_data, test_labels, c="g", s=6, label="Testing data")
#
#     if predictions is not None:
#         # Plot the predictions in red (predictions were made on the test data)
#         plt.scatter(test_data, predictions, c="r", s=6, label="Predictions")
#
#     # Show the legend
#     plt.legend(prop={"size": 10})
#     plt.show()
#
#
# plot_predictions()
