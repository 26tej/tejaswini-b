import numpy as np
import pandas as pd
data=pd.read_csv('C:/Users/saiga/Downloads/GOOG (1).csv')

df=data.reset_index()['close']
df

import matplotlib.pyplot as plt
plt.plot(df)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))
df.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert data to sequences
sequence_length = 10 
X, y = [], []
for i in range(len(df) - sequence_length):
    X.append(df[i:i+sequence_length])
    y.append(df[i+sequence_length])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history=model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)

from sklearn.metrics import mean_squared_error

# Predict forex values
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# Plot actual vs. predicted values
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Currency')
plt.title("google stock prices")
plt.legend()
plt.show()