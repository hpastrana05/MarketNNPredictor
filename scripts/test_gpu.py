import torch
import torch.nn as nn
import torch.optim as optim

# 1. Force GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Create a Simple Linear Model
# A single layer: 1 input -> 1 output
model = nn.Linear(1, 1).to(device)

# 3. Dummy Data (y = 2x)
# Creating 100 points
x_train = torch.randn(100, 1).to(device)
y_true = x_train * 2

# 4. Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Training Loop
for epoch in range(100):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_true)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print("\nModel trained. Weight should be close to 2.0:")
print(model.weight.item())