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
    y_df = pd.read_csv('dataset/y_train.csv')

    # Convert grid representations to numpy arrays
    x_time_df['env_input'] = x_time_df['env_input'].apply(lambda x: np.array(eval(x)))
    y_df['next_env'] = y_df[['val_x', 'val_y']].apply(lambda x: np.array(x), axis=1)

    # Merge dataframes on Integer time step
    merged_df = pd.merge(x_time_df, y_df, on='time_stamp')

    # Convert input of nested list to nested list 
    x_time = np.array(merged_df["env_input"].tolist())

    # Flatten nested list to single feature vector 
    x_time_flat = x_time.reshape(x_time.shape[0], -1)

    # Normalize features using MinMaxScaler
    scaler_x_time = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_time_scaled = scaler_x_time.fit_transform(x_time_flat)
    y_scaled = scaler_y.fit_transform(np.vstack(merged_df["next_env"].tolist()))

    # Convert to PyTorch tensors
    x_time_tensor = torch.tensor(x_time_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Reshape tensors for LSTM input
    sequence_length = 1  # Assuming sequence length is 1
    x_time_tensor = x_time_tensor.view(-1, sequence_length, x_time_tensor.shape[1])
    y_tensor = y_tensor[sequence_length - 1:]

    # Initialize model
    input_size = x_time_tensor.shape[2]
    lstm_hidden_size = 50
    model = CustomLSTMModel(input_size=input_size, lstm_hidden_size=lstm_hidden_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_time_tensor)
        loss = criterion(outputs.view(-1), y_tensor.view(-1))
        loss.backward()
        optimizer.step()
        
        # Print training statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm.pth')
    

# Example usage
train(sequence_length=2)  # Adjust sequence_length as needed
