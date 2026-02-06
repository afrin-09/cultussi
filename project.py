# ============================================
# ADVANCED TIME SERIES FORECASTING PROJECT
# Transformer with Attention Mechanism
# ============================================

# Uncomment if libraries are not installed
# !pip install numpy pandas scipy scikit-learn matplotlib torch statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm

# --------------------------------------------
# 1. Synthetic Multivariate Time Series Dataset
# --------------------------------------------
np.random.seed(42)

def generate_data(n_steps=2000):
    t = np.arange(n_steps)

    seasonal_1 = 2 * np.sin(2 * np.pi * t / 24)
    seasonal_2 = 1.5 * np.sin(2 * np.pi * t / 168)

    trend = 0.0005 * (t ** 2) / n_steps

    exog_1 = np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.2, n_steps)
    exog_2 = (t % 100 < 10).astype(float)

    y = seasonal_1 + seasonal_2 + trend + 0.8 * exog_1 + 0.5 * exog_2
    y += np.random.normal(0, 0.3, n_steps)

    return pd.DataFrame({
        "y": y,
        "seasonal_1": seasonal_1,
        "seasonal_2": seasonal_2,
        "exog_1": exog_1,
        "exog_2": exog_2
    })

data = generate_data()

# --------------------------------------------
# 2. Scaling and Windowing
# --------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

def create_windows(data, input_len=48, horizon=12):
    X, y = [], []
    for i in range(len(data) - input_len - horizon):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+horizon, 0])
    return np.array(X), np.array(y)

X, y = create_windows(scaled_data)

# --------------------------------------------
# 3. Train-Test Split
# --------------------------------------------
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# --------------------------------------------
# 4. Transformer Model with Attention
# --------------------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, horizon=12):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

model = TimeSeriesTransformer(input_dim=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------
# 5. Train Transformer Model
# --------------------------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

# --------------------------------------------
# 6. Evaluate Transformer Model
# --------------------------------------------
model.eval()
preds, actual = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb).numpy())
        actual.append(yb.numpy())

preds = np.vstack(preds)
actual = np.vstack(actual)

rmse_transformer = mean_squared_error(actual, preds, squared=False)
mae_transformer = mean_absolute_error(actual, preds)

# --------------------------------------------
# 7. ARIMA Baseline Model
# --------------------------------------------
train_y = data["y"][:split]
test_y = data["y"][split:split+len(y_test)]

arima_model = sm.tsa.ARIMA(train_y, order=(2,1,2))
arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(steps=len(test_y))

rmse_arima = mean_squared_error(test_y, arima_forecast, squared=False)
mae_arima = mean_absolute_error(test_y, arima_forecast)

# --------------------------------------------
# 8. Results Table
# --------------------------------------------
results = pd.DataFrame({
    "Model": ["ARIMA", "Transformer"],
    "RMSE": [rmse_arima, rmse_transformer],
    "MAE": [mae_arima, mae_transformer]
})

print("\nFinal Forecasting Metrics:\n")
print(results)

# --------------------------------------------
# 9. Plot: Actual vs Transformer Predictions
# --------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(actual[:200, 0], label="Actual")
plt.plot(preds[:200, 0], label="Transformer Prediction")
plt.title("Actual vs Transformer Forecast (First Horizon Step)")
plt.xlabel("Time Steps")
plt.ylabel("Scaled Value")
plt.legend()
plt.show()

# --------------------------------------------
# 10. Plot: ARIMA vs Actual
# --------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(test_y.values[:200], label="Actual")
plt.plot(arima_forecast.values[:200], label="ARIMA Forecast")
plt.title("Actual vs ARIMA Forecast")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()
