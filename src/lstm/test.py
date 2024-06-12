import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = 'your_file_path_here.csv'
df = pd.read_csv(file_path)

# Define cutoff year for train-test split
cutoff_year = df["Year"].max() - 1

# Split the data into training and testing sets
train_data = df[df["Year"] <= cutoff_year]
test_data = df[df["Year"] > cutoff_year]

# Select the target column
target_column = "Realized"

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform both train and test data
scaled_train_data = scaler.fit_transform(train_data.drop(columns=["Year", "Region"]))
scaled_test_data = scaler.transform(test_data.drop(columns=["Year", "Region"]))

# Convert the scaled data back to a DataFrame for easier manipulation
scaled_train_df = pd.DataFrame(scaled_train_data, columns=train_data.columns.drop(["Year", "Region"]))
scaled_test_df = pd.DataFrame(scaled_test_data, columns=test_data.columns.drop(["Year", "Region"]))


# Create sequences for LSTM
def create_sequences(data, target_column, sequence_length=10):
    sequences = []
    targets = []
    data_length = len(data)

    for i in range(data_length - sequence_length):
        seq = data.iloc[i:i + sequence_length].drop(columns=[target_column]).values
        target = data.iloc[i + sequence_length][target_column]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


sequence_length = 10

X_train, y_train = create_sequences(scaled_train_df, target_column, sequence_length)
X_test, y_test = create_sequences(scaled_test_df, target_column, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, X_train.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')