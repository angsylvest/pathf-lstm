import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from lstm import CustomLSTMModel

def train(sequence_length): 
    # Load CSV files
    x_time_df = pd.read_csv('dataset/x_time_train.csv')
    x_context_df = pd.read_csv('dataset/x_context_train.csv')
    y_df = pd.read_csv('dataset/y_train.csv')

    # Merge dataframes on Timestamp
    merged_df = pd.merge(x_time_df, x_context_df, on='time_stamp')
    merged_df = pd.merge(merged_df, y_df, on='time_stamp')

    # Convert Timestamp to datetime type
    merged_df['time_stamp'] = pd.to_datetime(merged_df['time_stamp'])

    # Sort by Timestamp
    merged_df.sort_values(by='time_stamp', inplace=True)

    # Extract features and target variable
    x_time = merged_df[['current_position_x', 'current_position_y', 'goal_position_x', 'goal_position_y']].values
    
    # Convert string representation to nested list of tuples
    merged_df['env_representation'] = merged_df['env_representation'].apply(lambda x: eval(x))
    x_context_array = np.array(merged_df['env_representation'].tolist())
    
    # Flatten the nested list to a single feature vector
    x_context_flat = x_context_array.reshape(x_context_array.shape[0], -1)

    # Normalize features using MinMaxScaler
    scaler_x_time = MinMaxScaler()
    scaler_x_context = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_time_scaled = scaler_x_time.fit_transform(x_time)
    x_context_scaled = scaler_x_context.fit_transform(x_context_flat)
    
    # Extract target variable
    y = merged_df[['next_position_x', 'next_position_y']].values
    y_scaled = scaler_y.fit_transform(y)

    # Convert to PyTorch tensors
    x_time_tensor = torch.tensor(x_time_scaled, dtype=torch.float32)
    x_context_tensor = torch.tensor(x_context_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Reshape tensors for LSTM input
    x_time_tensor = x_time_tensor.view(-1, sequence_length, x_time_tensor.shape[1])
    x_context_tensor = x_context_tensor.view(-1, sequence_length, x_context_tensor.shape[1])
    y_tensor = y_tensor[sequence_length - 1:]

    print(f'features sizes: {x_time_tensor.shape[2]} {x_context_tensor.shape[2]}')

    # Initialize model
    model = CustomLSTMModel(num_time_features=x_time_tensor.shape[2], num_context_features=x_context_tensor.shape[2], lstm_hidden_size=50)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 50

    # Training loop
    for epoch in range(num_epochs):
        # Set the model in training mode
        model.train()

        # Forward pass
        outputs = model(x_time_tensor, x_context_tensor)

        loss = criterion(outputs, y_tensor.view(-1, 2))  # Assuming y_tensor is a 2D tensor with (next_position_x, next_position_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm.pth')

# Example usage
train(sequence_length=2)  # Adjust sequence_length as needed
