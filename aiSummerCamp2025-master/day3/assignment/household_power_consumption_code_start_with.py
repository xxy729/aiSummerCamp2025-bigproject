# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data\\household_power_consumption.txt', sep = ";")
df.head()
# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler

feature_cols = [col for col in train.columns if col not in ['datetime', 'Global_active_power']]
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# %%
# split X and y
# 以'Global_active_power'为预测目标
X_train = train[feature_cols].values
y_train = train['Global_active_power'].values
X_test = test[feature_cols].values
y_test = test['Global_active_power'].values

# %%
# create dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
# build a LSTM model
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = LSTMModel(input_size=X_train.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# %%
# train the model
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# %%
# evaluate the model on the test set
model.eval()
preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds.extend(output.cpu().numpy())

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.4f}")

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(y_test[:200], label='True')
plt.plot(preds[:200], label='Predicted')
plt.legend()
plt.title('LSTM Prediction vs Ground Truth')
plt.show()
