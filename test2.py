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
# print(X[:5], y[:5])
# print(X.shape, y.shape)

# Prepare train data and test data
train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
# print(f"{train_split}")
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
# print(X_test, X_train)
# print(f"{len(X_train)}, {len(y_train)}, {len(X_test)}, {len(y_test)}")


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


# Define the model
class NextNumberPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NextNumberPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    # Train the model
    def train(model, data, loss_fn, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for input_sequence, target in data:
                input_sequence = torch.Tensor(input_sequence).view(len(input_sequence), 1, -1)
                target = torch.Tensor(target).view(len(target), -1)

                # Forward pass
                output = model(input_sequence)
                loss = loss_fn(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Test the model
    def test(model, data, loss_fn):
        total_loss = 0
        for input_sequence, target in data:
            input_sequence = torch.Tensor(input_sequence).view(len(input_sequence), 1, -1)
            target = torch.Tensor(target).view(len(target), -1)
            output = model(input_sequence)
            total_loss += loss_fn(output, target).item()
        return total_loss / len(data)


# Setup the model, data, loss function and optimizer
model = NextNumberPredictor(1, 32, 1)
data = [(list(range(10)), list(range(1, 11))), (list(range(10, 20)), list(range(11, 21)))]
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
train(model, data, loss_fn, optimizer, num_epochs=100)

# Use the model to make predictions
input_sequence = torch.Tensor(list(range(10, 20))).view(10, 1, -1)
output = model(input_sequence)
prediction = output[-1].item()
print(f'Predicted next number: {prediction:.4f}')




#
# # Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
