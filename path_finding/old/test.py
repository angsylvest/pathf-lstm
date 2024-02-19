import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm import CustomLSTMModel

# Example input data for testing
x_time_data = {
    'current_position_x': [5],
    'current_position_y': [10],
    'goal_position_x': [20],
    'goal_position_y': [30],
    'time_stamp': ['2024-01-01 08:00:00']
}

x_context_data = {
    'env_representation': ['[[(5, 3, 10, 15), (0, 0, 0, 0), (8, 20, 25, 20)], [(0,0,0,0), (0,0,0,0), (0,0,0,0)], [(0,0,0,0), (0,0,0,0), (0,0,0,0)]]'],
    'time_stamp': ['2024-01-01 08:00:00']
}

# Convert to pandas DataFrame
x_time_df = pd.DataFrame(x_time_data)
x_context_df = pd.DataFrame(x_context_data)

# Convert Timestamp to datetime type
x_time_df['time_stamp'] = pd.to_datetime(x_time_df['time_stamp'])

# Extract features
x_time = x_time_df[['current_position_x', 'current_position_y', 'goal_position_x', 'goal_position_y']].values

# Convert string representation to nested list of tuples
x_context_df['env_representation'] = x_context_df['env_representation'].apply(eval)
x_context_array = np.array(x_context_df['env_representation'].tolist())

# Flatten the nested list to a single feature vector
x_context_flat = x_context_array.reshape(x_context_array.shape[0], -1)

# Normalize features using MinMaxScaler
scaler_x_time = MinMaxScaler()
scaler_x_context = MinMaxScaler()

x_time_scaled = scaler_x_time.fit_transform(x_time)
x_context_scaled = scaler_x_context.fit_transform(x_context_flat)

# Convert to PyTorch tensors
x_time_tensor = torch.tensor(x_time_scaled, dtype=torch.float32)
x_context_tensor = torch.tensor(x_context_scaled, dtype=torch.float32)

# Reshape tensors for LSTM input
# sequence_length = 1
x_time_tensor = x_time_tensor.unsqueeze(0)  # Add batch dimension
x_context_tensor = x_context_tensor.unsqueeze(0)  # Add batch dimension

print(f'x_time_tensor {x_time_tensor.shape} and x_context_tensor {x_context_tensor.shape}')

# Load the trained model
model = CustomLSTMModel(num_time_features=x_time_tensor.shape[2], num_context_features=x_context_tensor.shape[2], lstm_hidden_size=50)
model.load_state_dict(torch.load('lstm.pth'))
model.eval()

# Perform inference
with torch.no_grad():
    predicted_output = model(x_time_tensor, x_context_tensor)

# Convert predicted output to numpy array
predicted_output = predicted_output.numpy()

# Print or use the predicted output as needed
print(predicted_output)
