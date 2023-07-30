# PyTorch documentation
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)
# print(np.__version__)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# Import X and y from csv files
X = torch.from_numpy(np.genfromtxt("results_X.csv", delimiter=",", dtype="int16"))
y = torch.from_numpy((np.genfromtxt("results_y.csv", delimiter=",", dtype="int16")))
print(X[:5], y[:5])

# Prepare train data and test data
train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
# print(f"{train_split}")
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(f"{len(X_train)}, {len(y_train)}, {len(X_test)}, {len(y_test)}")


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):

    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(8, 6))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=6, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=6, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=6, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 10})
    plt.show()


plot_predictions()

# class TestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(TestModel, self).__init__()
#
#         self.linear1 = torch.nn.Linear(100, 200)
#         self.activation = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(200, 10)
#         self.softmax = torch.nn.Softmax()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#         return x


# test_model = TestModel()

# print('The model:')
# print(test_model)


# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
    # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    # Forward defines the computation in the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
