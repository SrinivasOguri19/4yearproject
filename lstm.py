import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score


# Read the dataset from the CSV file
dataset = pd.read_csv('finaldata.csv')

# Extract the input features and target variable
X = dataset.drop(['energy_kWh after normalisation',"date", "energy_kWh"], axis=1).values
y = dataset['energy_kWh after normalisation'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the training and testing data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the stacked LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=7, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Testing loss:", loss)
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
score = model.score(X_test, y_test)
# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared: ", r2)