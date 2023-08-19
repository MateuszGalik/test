import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

data_size = 90

X = (torch.from_numpy(np.genfromtxt("results_X.csv", delimiter=",", dtype="float32")).unsqueeze(dim=1))[data_size:]
y_L15 = (torch.from_numpy((np.genfromtxt("results_y_L15.csv", delimiter=",", dtype="float32"))))[data_size:]
train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
# # X = X[:20])
# print(X)
# # y_L15 = y_L15[:20]
# print(y_L15)

X_train = X[:train_split]
y_L15_train = y_L15[:train_split]
X_test = X[train_split:]
y_L15_test = y_L15[train_split:]

print(X_test)
# print(X_train[:3])
# print(y_L15_train[:3])

def plot_prediction(train_data=None,
                    train_labels=None,
                    test_data=None,
                    test_labels=None,
                    pred_data=None,
                    pred_labels=None,
                    learned_data=None,
                    learned_labels=None):
    plt.figure(figsize=(10, 6))
    colors = [["blue"], ["orange"], ["green"], ["red"], ["purple"]]
    tr_labels = ["tr_L1", "tr_L2", "tr_L3", "tr_L4", "_trL5"]
    tr_colors = colors
    te_labels = ["te_L1", "te_L2", "te_L3", "te_L4", "te_L5"]
    te_colors = colors
    pr0_labels = ["p0_L1", "p0_L2", "p0_L3", "p0_L4", "p0_L5"]
    pl0_marker = [".",".",".",".",".",]
    pl0_colors=colors
    pl_labels = ["pl_L1", "pl_L2", "pl_L3", "pl_L4", "pl_L5"]
    pl_marker = ["x","x","x","x","x"]
    pl_colors=colors

    if train_data is not None:
        for tr_y, tr_labels,tr_colors in zip(np.array(train_labels).T, tr_labels, tr_colors):
            plt.scatter(train_data, tr_y, label=tr_labels, c=tr_colors)

    if test_data is not None:
        for te_y, te_labels, te_colors in zip(np.array(test_labels).T, te_labels, te_colors):
            plt.scatter(test_data, te_y, label=te_labels, c=te_colors)

    if pred_data is not None:
        for pr0_y, pr0_labels ,pl0_marker, pl0_colors in zip(np.array(pred_labels).T, pr0_labels, pl0_marker, pl0_colors):
            plt.scatter(pred_data, pr0_y, label=pr0_labels, marker=pl0_marker, c=pl0_colors)

    if learned_data is not None:
        for pl_y, pl_labels, pl_marker, pl_colors in zip(np.array(learned_labels).T, pl_labels, pl_marker, pl_colors):
            plt.scatter(learned_data, pl_y, label=pl_labels, marker=pl_marker, c=pl_colors )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# plot_prediction(train_data=X_train,
#                 train_labels=y_L15_train,
#                 test_data=X_test,
#                 test_labels=y_L15_test)


class NoLinearModel(nn.Module):

    def __init__(self):
        super().__init__()

        # self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True)
        # self.linear = nn.Linear(50, 5)

        self.layer_1 = nn.Linear(in_features=5, out_features=25)
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        # self.layer_2 = nn.Linear(in_features=25, out_features=1)
        # self.lstm2 = nn.LSTM(input_size=1, hidden_size=25)
        self.layer_3 = nn.Linear(in_features=100, out_features=5)
        # self.layer_4 = nn.Linear(in_features=25, out_features=25)
        # self.layer_5 = nn.Linear(in_features=25, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

        # self.linear1 = torch.nn.Linear(1, 200)
        # self.activation = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(200, 5)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # return self.sigmoid(self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))))
        x, _ = self.lstm(x)
        # x = self.sigmoid(self.layer_2(x))
        x, _ = self.lstm2(x)
        # x = self.layer_3(x)
        x = self.layer_3(x)
        # x = self.softmax(x)

        # x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.softmax(x)
        return x


torch.manual_seed(42)
model_L15 = NoLinearModel()
with torch.inference_mode():
    y_L15_preds = model_L15(X_test)
    # print(y_L15_preds[:3])

plot_prediction(train_data=X_train,
                train_labels=y_L15_train,
                test_data=X_test,
                test_labels=y_L15_test,
                pred_data=X_test,
                pred_labels=y_L15_preds)

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
# optimizer = torch.optim.SGD(params=model_L15.parameters(), lr=0.01)
optimizer = torch.optim.Adam(params=model_L15.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(params=model_L15.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(params=model_L15.parameters(), lr=0.001)

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 1000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training
    # Put model in training mode (this is the default state of a model)
    model_L15.train()
    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_L15(X_train)
    # print(y_pred)
    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_L15_train)
    # 3. Zero grad of the optimizer
    optimizer.zero_grad()
    # 4. Loss backwards
    loss.backward()
    # 5. Progress the optimizer
    optimizer.step()
    ### Testing
    # Put the model in evaluation mode
    model_L15.eval()
    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_L15(X_test)
      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_L15_test.type(torch.float))
      # Print out what's happening
      if epoch % 50 == 0:
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

# Turn model into evaluation mode
model_L15.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_L15(X_test)
# print(y_preds)


plot_prediction(train_data=X_train,
                train_labels=y_L15_train,
                test_data=X_test,
                test_labels=y_L15_test,
                # pred_data=X_test,
                # pred_labels=y_L15_preds,
                learned_data=X_test,
                learned_labels=y_preds)


# Use the model to make predictions
# input_sequence = torch.Tensor(list(range(10, 20))).view(10, 1, -1)
input_sequence = torch.Tensor(X_test)
output_L15 = model_L15(input_sequence)
print(output_L15)
# prediction_L15 = output_L15[:1]
# print(f'Predicted next number L15: {prediction_L15:.2f}')