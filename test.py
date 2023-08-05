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


# plot_predictions()
class LinearRegressionModel(nn.Module):
    # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1,)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


m = nn.Linear(20, 30)
inputL = torch.randn(128, 20)
outputL = m(inputL)
print(outputL.size())

rnn = nn.RNN(10, 20, 2)
inputR = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
outputR, hn = rnn(inputR, h0)
print(outputR.size())

class LinearRegressionModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


class NoLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=25)
        self.layer_2 = nn.Linear(in_features=25, out_features=25)
        self.layer_3 = nn.Linear(in_features=25, out_features=25)
        self.layer_4 = nn.Linear(in_features=25, out_features=25)
        self.layer_5 = nn.Linear(in_features=25, out_features=1)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))))


class NoLinearModel2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(NoLinearModel2, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


torch.manual_seed(42)


model_0 = NoLinearModel()

# print("### Model 0 Parameters")
# print(list(model_0.parameters()))
# print(model_0.state_dict())


# Make predictions with model
with torch.inference_mode():
    # y_preds = model_0(X_test)
    y_preds = model_0(X_test)
    # y_preds = y[train_split:]

plot_predictions(predictions=y_preds)

# Create the loss function
loss_fn = nn.L1Loss()  # train loss 4.06, test loss 3.8
# loss_fn = nn.MSELoss() # train loss 28.1, test loss 23.52
# loss_fn = nn.CrossEntropyLoss() # train loss 0.0, test loss 0.0
# loss_fn = nn.CTCLoss() # lack of parameters
# loss_fn = nn.NLLLoss() # lack of parameters
# loss_fn = nn.PoissonNLLLoss() # train loss -6.4, test loss -6.7
# loss_fn = nn.GaussianNLLLoss() # lack of parameters
# loss_fn =  nn.KLDivLoss() # train loss -4674, test loss -4762
# loss_fn = nn.BCELoss() # lack of parameters
# loss_fn = nn.BCEWithLogitsLoss() # train loss -3428, test loss -3503
# loss_fn = nn.MarginRankingLoss() # lack of parameters
# loss_fn = nn.HingeEmbeddingLoss() # train loss 0.36, test loss 0.37
# loss_fn = nn.MultiLabelMarginLoss() # lack of parameters
# loss_fn = nn.HuberLoss() # train loss 3.5, test loss 3.4
# loss_fn = nn.SmoothL1Loss() # train loss 3.5, test loss 3.4
# loss_fn = nn.SoftMarginLoss() # train los 0.01, test loss 0.01
# loss_fm = nn.MultiLabelSoftMarginLoss() # lack of parameters
# loss_fm = nn.CosineEmbeddingLoss() # lack of parameters
# loss_fn = nn.MultiMarginLoss() # lack of parameters
# loss_fn = nn.TripletMarginLoss() # lack of parameters
# less_fn = nn.TripletMarginWithDistanceLoss() # lack of parameters

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
