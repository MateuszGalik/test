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
y_L1 = torch.from_numpy((np.genfromtxt("results_y_L1.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)
y_L2 = torch.from_numpy((np.genfromtxt("results_y_L2.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)
y_L3 = torch.from_numpy((np.genfromtxt("results_y_L3.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)
y_L4 = torch.from_numpy((np.genfromtxt("results_y_L4.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)
y_L5 = torch.from_numpy((np.genfromtxt("results_y_L5.csv", delimiter=",", dtype="float32"))).unsqueeze(dim=1)

# print(X[3:], y_L1[:3], y_L2[:3], y_L3[:3], y_L4[:3], y_L5[:3])
# print(X.shape, y.shape)

# Prepare train data and test data
train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
# print(f"{train_split}")
X_train = X[:train_split]
y_L1_train = y_L1[:train_split]
y_L2_train = y_L2[:train_split]
y_L3_train = y_L3[:train_split]
y_L4_train = y_L4[:train_split]
y_L5_train = y_L5[:train_split]

X_train = X_train/6000
X_test = X[train_split:]
X_test = X_test/6000
y_L1_test = y_L1[train_split:]
y_L2_test = y_L2[train_split:]
y_L3_test = y_L3[train_split:]
y_L4_test = y_L4[train_split:]
y_L5_test = y_L5[train_split:]

# print(X_test[:5], X_train[:5])
# print(f"{len(X_train)}, {len(y_train)}, {len(X_test)}, {len(y_test)}")


def plot_predictions(train_data=X_train,
                     train_l1_labels=y_L1_train,
                     train_l2_labels=y_L2_train,
                     train_l3_labels=y_L3_train,
                     train_l4_labels=y_L4_train,
                     train_l5_labels=y_L5_train,
                     test_data=X_test,
                     test_l1_labels=y_L1_test,
                     test_l2_labels=y_L2_test,
                     test_l3_labels=y_L3_test,
                     test_l4_labels=y_L4_test,
                     test_l5_labels=y_L5_test,
                     predictions_l1=None,
                     predictions_l2=None,
                     predictions_l3=None,
                     predictions_l4=None,
                     predictions_l5=None):

    """
    Plots training data, test data and compares predictions.
    """
    # plt.figure(figsize=(8, 6))
    fig, (ax) = plt.subplots(figsize=(15, 6))

    # Plot training data in blue
    ax.plot(train_data, train_l1_labels, linestyle='none', color="blue", marker='o', markersize=2, label="L1")
    ax.plot(train_data, train_l2_labels, linestyle='none', color="orange", marker='o', markersize=2, label="L2")
    ax.plot(train_data, train_l3_labels, linestyle='none', color="green", marker='o', markersize=2, label="L3")
    ax.plot(train_data, train_l4_labels, linestyle='none', color="gray", marker='o', markersize=2, label="L4")
    ax.plot(train_data, train_l5_labels, linestyle='none', color="purple", marker='o', markersize=2, label="L5")

    # Plot test data
    ax.plot(test_data, test_l1_labels, linestyle='none', color="blue", marker='x', markersize=3, label="L1_test")
    ax.plot(test_data, test_l2_labels, linestyle='none', color="orange", marker='x', markersize=3, label="L2_test")
    ax.plot(test_data, test_l3_labels, linestyle='none', color="green", marker='x', markersize=3, label="L3_test")
    ax.plot(test_data, test_l4_labels, linestyle='none', color="gray", marker='x', markersize=3, label="L4_test")
    ax.plot(test_data, test_l5_labels, linestyle='none', color="purple", marker='x', markersize=3, label="L5_test")

    # # Plot the predictions of L1 to L5 if exists
    if predictions_l1 is not None:
        ax.plot(test_data, predictions_l1, linestyle='none', color="blue", marker='p', markersize=4, label="L1_pred")
    if predictions_l2 is not None:
        ax.plot(test_data, predictions_l2, linestyle='none', color="orange", marker='p', markersize=4, label="L2_pred")
    if predictions_l3 is not None:
        ax.plot(test_data, predictions_l3, linestyle='none', color="green", marker='p', markersize=4, label="L3_pred")
    if predictions_l4 is not None:
        ax.plot(test_data, predictions_l4, linestyle='none', color="gray", marker='p', markersize=4, label="L4_pred")
    if predictions_l5 is not None:
        ax.plot(test_data, predictions_l5, linestyle='none', color="purple", marker='p', markersize=4, label="L5_pred")

    # Show the legend
    ax.legend(prop={"size": 10})
    # Y axis limitations
    plt.ylim([0, 45])
    plt.show()

# ########################################################################################################
# # plot_predictions()
#         return self.linear_layer(x)
#
#
# class LinearRegressionModel(nn.Module):
#     # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
#     def __init__(self):
#         super().__init__()
#         # Use nn.Linear() for creating the model parameters
#         self.linear_layer = nn.Linear(in_features=1, out_features=1,)
#
#     # Define the forward computation (input data x flows through nn.Linear())
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # randomtest = torch.randn(2, 3 , 4)
# # print(f"Random Test: {randomtest[:5]}")
# #
# # m = nn.Linear(1, 2)
# # inputL = torch.randn(128, 1)
# # outputL = m(inputL)
# # print(outputL.size())
# #
# # print(X_test.size())
# # rnn = nn.RNN(1, 200, 1)
# # inputR = torch.randn(200, 1)
# # print(f"InputR size: {inputR.size()}")
# # # print(f"InputR: {inputR}")
# # h0 = torch.randn(1, 200)
# # print(f"h0: {h0}")
# # outputR, hn = rnn(inputR, h0)
# # print(f"Output size{outputR.size()}, Hn size: {hn.size()}")
# # print(outputR)
#
# # Ploting know x and random y  to yn
# # import torch
# # import matplotlib.pyplot as plt
# # x = torch.arange(start=0, end=200, step=1)
# # y = torch.randn(200, 3)
# # torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
# # plt.plot(x, y)  # Works now
# # plt.show()
#
# class LinearRegressionModel2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
#
#     # Forward defines the computation in the model
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.weights * x + self.bias


class NoLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=25)
        self.lstm = nn.LSTM(input_size=1, hidden_size=25)
        self.layer_2 = nn.Linear(in_features=25, out_features=1)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=25)
        self.layer_3 = nn.Linear(in_features=25, out_features=1)
        # self.layer_4 = nn.Linear(in_features=25, out_features=25)
        # self.layer_5 = nn.Linear(in_features=25, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))))
        x, _ = self.lstm(x)
        # x = self.sigmoid(self.layer_2(x))
        # x, _ = self.lstm2(x)
        # x = self.layer_3(x)
        x = self.layer_3(x)
        return x
#
#
# class NoLinearModel2(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(NoLinearModel2, self).__init__()
#
#         # Defining some parameters
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#
#         # Defining the layers
#         # RNN Layer
#         self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim, output_size)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         # Initializing hidden state for first input using method defined below
#         hidden = self.init_hidden(batch_size)
#
#         # Passing in the input and hidden state into the model and obtaining outputs
#         out, hidden = self.rnn(x, hidden)
#
#         # Reshaping the outputs such that it can be fit into the fully connected layer
#         out = out.contiguous().view(-1, self.hidden_dim)
#         out = self.fc(out)
#
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         # This method generates the first hidden state of zeros which we'll use in the forward pass
#         # We'll send the tensor holding the hidden state to the device we specified earlier as well
#         hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         return hidden

#############################################################################################

# Define the model


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


torch.manual_seed(42)

# model_0 = NoLinearModel2(input_size=1, output_size=1, hidden_dim=200, n_layers=1)
# model_0 = LSTMmodel(1, 32, 1)
model_L1 = NoLinearModel()
model_L2 = NoLinearModel()
model_L3 = NoLinearModel()
model_L4 = NoLinearModel()
model_L5 = NoLinearModel()

# print("### Model 0 Parameters")
# print(list(model_0.parameters()))
# print(model_0.state_dict())
# print(X_test.T.size())

# Make predictions with model
with torch.inference_mode():
    # y_preds, hn = model_0(X_test.unsqueeze(dim=1))
    y_L1_preds = model_L1(X_test)
    y_L2_preds = model_L2(X_test)
    y_L3_preds = model_L2(X_test)
    y_L4_preds = model_L2(X_test)
    y_L5_preds = model_L2(X_test)
    # y_preds = y[train_split:]
# print(y_L1_preds)
plot_predictions(predictions_l1=y_L1_preds,
                 predictions_l2=y_L2_preds,
                 predictions_l3=y_L3_preds,
                 predictions_l4=y_L4_preds,
                 predictions_l5=y_L5_preds)

# Create the loss function
# loss_fn = nn.L1Loss()  # train loss 4.06, test loss 3.8
loss_fn = nn.MSELoss()  # train loss 28.1, test loss 23.52
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
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01, momentum=0.9)
optimizer_L1 = torch.optim.Adam(model_L1.parameters())
optimizer_L2 = torch.optim.Adam(model_L2.parameters())
optimizer_L3 = torch.optim.Adam(model_L3.parameters())
optimizer_L4 = torch.optim.Adam(model_L4.parameters())
optimizer_L5 = torch.optim.Adam(model_L5.parameters())

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
# epochs = 2200
epochs = 1500
# Create empty loss lists to track values
train_loss_values_L1 = []
test_loss_values_L1 = []
epoch_count_L1 = []

train_loss_values_L2 = []
test_loss_values_L2 = []
epoch_count_L2 = []

train_loss_values_L3 = []
test_loss_values_L3 = []
epoch_count_L3 = []

train_loss_values_L4 = []
test_loss_values_L4 = []
epoch_count_L4 = []

train_loss_values_L5 = []
test_loss_values_L5 = []
epoch_count_L5 = []

for epoch in range(epochs):
    # Training

    # Put model in training mode (this is the default state of a model)
    model_L1.train()
    model_L2.train()
    model_L3.train()
    model_L4.train()
    model_L5.train()

    # 1. Forward pass on train data using the forward() method inside
    y_L1_pred = model_L1(X_train)
    y_L2_pred = model_L2(X_train)
    y_L3_pred = model_L3(X_train)
    y_L4_pred = model_L4(X_train)
    y_L5_pred = model_L5(X_train)

    # y_pred, hn = model_0(X_train.unsqueeze(dim=1))
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss_L1 = loss_fn(y_L1_pred, y_L1_train)
    loss_L2 = loss_fn(y_L2_pred, y_L2_train)
    loss_L3 = loss_fn(y_L3_pred, y_L3_train)
    loss_L4 = loss_fn(y_L4_pred, y_L4_train)
    loss_L5 = loss_fn(y_L5_pred, y_L5_train)


    # 3. Zero grad of the optimizer
    optimizer_L1.zero_grad()
    optimizer_L2.zero_grad()
    optimizer_L3.zero_grad()
    optimizer_L4.zero_grad()
    optimizer_L5.zero_grad()

    # 4. Loss backwards
    loss_L1.backward()
    loss_L2.backward()
    loss_L3.backward()
    loss_L4.backward()
    loss_L5.backward()

    # 5. Progress the optimizer
    optimizer_L1.step()
    optimizer_L2.step()
    optimizer_L3.step()
    optimizer_L4.step()
    optimizer_L5.step()

    # Testing

    # Put the model in evaluation mode
    model_L1.eval()
    model_L2.eval()
    model_L3.eval()
    model_L4.eval()
    model_L5.eval()

    with torch.inference_mode():

        # 1. Forward pass on test data
        # test_pred, hnt = model_0(X_test.unsqueeze(dim=1))
        test_pred_L1 = model_L1(X_test)
        test_pred_L2 = model_L2(X_test)
        test_pred_L3 = model_L3(X_test)
        test_pred_L4 = model_L4(X_test)
        test_pred_L5 = model_L5(X_test)

        # 2. Caculate loss on test data
        test_loss_L1 = loss_fn(test_pred_L1, y_L1_test.type(torch.float))
        test_loss_L2 = loss_fn(test_pred_L2, y_L2_test.type(torch.float))
        test_loss_L3 = loss_fn(test_pred_L3, y_L3_test.type(torch.float))
        test_loss_L4 = loss_fn(test_pred_L4, y_L4_test.type(torch.float))
        test_loss_L5 = loss_fn(test_pred_L5, y_L5_test.type(torch.float))

        # Print out what's happening
        if epoch % 100 == 0:
            epoch_count_L1.append(epoch)
            epoch_count_L2.append(epoch)
            epoch_count_L3.append(epoch)
            epoch_count_L4.append(epoch)
            epoch_count_L5.append(epoch)
            train_loss_values_L1.append(loss_L1.detach().numpy())
            train_loss_values_L2.append(loss_L2.detach().numpy())
            train_loss_values_L3.append(loss_L3.detach().numpy())
            train_loss_values_L4.append(loss_L4.detach().numpy())
            train_loss_values_L5.append(loss_L5.detach().numpy())
            test_loss_values_L1.append(test_loss_L1.detach().numpy())
            test_loss_values_L2.append(test_loss_L2.detach().numpy())
            test_loss_values_L3.append(test_loss_L3.detach().numpy())
            test_loss_values_L4.append(test_loss_L4.detach().numpy())
            test_loss_values_L5.append(test_loss_L5.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss L1: {loss_L1} | MAE Test Loss L1: {test_loss_L1} ")
            print(f"Epoch: {epoch} | MAE Train Loss L2: {loss_L2} | MAE Test Loss L2: {test_loss_L2} ")
            print(f"Epoch: {epoch} | MAE Train Loss L3: {loss_L3} | MAE Test Loss L3: {test_loss_L3} ")
            print(f"Epoch: {epoch} | MAE Train Loss L4: {loss_L4} | MAE Test Loss L4: {test_loss_L4} ")
            print(f"Epoch: {epoch} | MAE Train Loss L5: {loss_L5} | MAE Test Loss L5: {test_loss_L5} ")


# Plot the loss curves
plt.plot(epoch_count_L1, train_loss_values_L1, label="Train loss")
plt.plot(epoch_count_L1, test_loss_values_L1, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
# print(model_0.state_dict())

model_L1.eval()
model_L2.eval()
model_L3.eval()
model_L4.eval()
model_L5.eval()

with torch.inference_mode():
  y_L1_preds = model_L1(X_test)
  y_L2_preds = model_L2(X_test)
  y_L3_preds = model_L3(X_test)
  y_L4_preds = model_L4(X_test)
  y_L5_preds = model_L5(X_test)
# print(y_L1_preds, y_L2_preds)


plot_predictions(predictions_l1=y_L1_preds,
                 predictions_l2=y_L2_preds,
                 predictions_l3=y_L3_preds,
                 predictions_l4=y_L4_preds,
                 predictions_l5=y_L5_preds)

# Use the model to make predictions
# input_sequence = torch.Tensor(list(range(10, 20))).view(10, 1, -1)
input_sequence = torch.Tensor(X_test)
output_L1 = model_L1(input_sequence)
output_L2 = model_L2(input_sequence)
output_L3= model_L3(input_sequence)
output_L4= model_L4(input_sequence)
output_L5= model_L5(input_sequence)

prediction_L1 = output_L1[-1].item()
prediction_L2 = output_L2[-1].item()
prediction_L3 = output_L3[-1].item()
prediction_L4 = output_L4[-1].item()
prediction_L5 = output_L5[-1].item()
print(f'Predicted next number L1: {prediction_L1:.2f}')
print(f'Predicted next number L2: {prediction_L2:.2f}')
print(f'Predicted next number L3: {prediction_L3:.2f}')
print(f'Predicted next number L4: {prediction_L4:.2f}')
print(f'Predicted next number L5: {prediction_L5:.2f}')