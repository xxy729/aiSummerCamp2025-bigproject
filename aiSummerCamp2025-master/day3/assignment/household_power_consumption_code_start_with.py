# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";", na_values=["?", "NA", "nan", "NaN"])
df.dropna(inplace=True)
df['Global_active_power'] = df['Global_active_power'].astype(float)
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
scaler = MinMaxScaler()
train_power = train[['Global_active_power']].values
test_power = test[['Global_active_power']].values
scaler.fit(train_power)
train_scaled = scaler.transform(train_power)
test_scaled = scaler.transform(test_power)
# %%
# split X and y
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

seq_length = 24  # 以24小时为一个序列
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)
# %%
# creat dataloaders
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = PowerDataset(X_train, y_train)
test_dataset = PowerDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# %%
# build a LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %%
# train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        X_batch = X_batch.view(-1, seq_length, 1)
        y_batch = y_batch.unsqueeze(1)  # 保证y和输出维度一致
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
# %%
# evaluate the model on the test set
model.eval()
preds = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        X_batch = X_batch.view(-1, seq_length, 1)
        output = model(X_batch)
        preds.append(output.cpu().numpy())
        actuals.append(y_batch.numpy())
preds = np.concatenate(preds).reshape(-1, 1)
actuals = np.concatenate(actuals).reshape(-1, 1)
preds_inv = scaler.inverse_transform(preds)
actuals_inv = scaler.inverse_transform(actuals)
mse = np.mean((preds_inv - actuals_inv) ** 2)
print(f"Test MSE: {mse:.4f}")
# %%
# plotting the predictions against the ground truth
plt.figure(figsize=(12, 6))
plt.plot(actuals_inv[:500], label='True')
plt.plot(preds_inv[:500], label='Predicted')
plt.legend()
plt.title('LSTM Prediction vs Ground Truth')
plt.show()

# %%
