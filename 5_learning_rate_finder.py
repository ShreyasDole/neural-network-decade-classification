import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from 2_model_architectures import create_model_1
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = X_train.pop('decade')

# Convert labels to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Create model and optimizer
model = create_model_1(X_train.shape[1], len(label_encoder.classes_))
optimizer = optim.SGD(model.parameters(), lr=1e-7)

# Implement learning rate finder
lrs = []
losses = []
for lr in torch.logspace(-7, 0, 100):
    optimizer.param_groups[0]['lr'] = lr.item()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = torch.nn.CrossEntropyLoss()(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    lrs.append(lr.item())
    losses.append(loss.item())

# Plot the learning rate vs loss
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.show()
