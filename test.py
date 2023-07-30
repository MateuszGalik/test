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
X = torch.from_numpy(np.genfromtxt("results_X.csv", delimiter=",", dtype="float32")).unsqueeze(dim=1)
y = torch.from_numpy((np.genfromtxt("results_y.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)
print(X[:5], y[:5])
print(X.shape, y.shape)

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


# plot_predictions()

class TestModel(torch.nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = torch.nn.Linear(1, 20)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 1)
        self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


# print('The model:')
# print(test_model)


# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
    # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1,)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class LinearRegressionModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
# model_0 = LinearRegressionModel()
model_0 = TestModel()
print(list(model_0.parameters()))

# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

# Create the loss function
loss_fn = nn.L1Loss()

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 1000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():

        # 1. Forward pass on test data
        test_pred = model_0(X_test)

        # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        # Print out what's happening
        if epoch % 100 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
