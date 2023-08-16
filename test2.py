import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.from_numpy(np.genfromtxt("results_X.csv", delimiter=",", dtype="float32")).unsqueeze(dim=1)
y_L15 = torch.from_numpy((np.genfromtxt("results_y_L15.csv", delimiter=",", dtype="float32")))
train_split = int(0.8 * len(X))  # 80% od data used for training set, 20% for testing
X_train = X[:train_split]
y_L15_train = y_L15[:train_split]
X_test = X[train_split:]
y_L15_test = y_L15[train_split:]


class NoLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=5, out_features=25)
        self.lstm = nn.LSTM(input_size=1, hidden_size=25)
        self.layer_2 = nn.Linear(in_features=25, out_features=1)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=25)
        self.layer_3 = nn.Linear(in_features=25, out_features=5)
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


torch.manual_seed(42)
model_L15 = NoLinearModel()
with torch.inference_mode():
    y_L15_preds = model_L15(X_test)
    print(y_L15_preds[:3])

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
# optimizer = torch.optim.SGD(params=model_L15.parameters(), lr=0.01)
optimizer = torch.optim.Adam(params=model_L15.parameters(), lr=0.01)

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
print(y_preds)
