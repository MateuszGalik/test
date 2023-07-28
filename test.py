# PyTorch documentation
import torch
print(torch.__version__)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
# without un-squeeze, errors will happen later on (shapes within linear layers)
X = torch.arange(start, end, step).unsqueeze(dim=1)
# print(X)
y = weight * X + bias
# print(f"{X[:5]}, {y[:5]}")

# TODO (create model, and plot)
