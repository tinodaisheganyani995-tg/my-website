import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Split into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)

# Normalize features
means = X_train.mean(dim=0, keepdims=True)
stds = X_train.std(dim=0, keepdims=True)

X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

# Convert targets to column vectors
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Create model parameters
torch.manual_seed(42)

n_features = X_train.shape[1]  # 8 features

w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

# Training loop
learning_rate = 0.4
n_epochs = 20

for epoch in range(n_epochs):

    # Forward pass
    y_pred = X_train @ w + b

    # Mean Squared Error loss
    loss = ((y_pred - y_train) ** 2).mean()

    # Compute gradients
    loss.backward()

    # Gradient descent step
    with torch.no_grad():
        b -= learning_rate * b.grad
        w -= learning_rate * w.grad

    # Reset gradients
    b.grad.zero_()
    w.grad.zero_()



    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

# Predictions on new data
X_new = X_test[:3]

with torch.no_grad():
    y_pred = X_new @ w + b

print("\nPredictions:")
print(y_pred)

      
        




