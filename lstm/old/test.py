from lstm import CustomLSTMModel
import torch 

import pandas as pd
import numpy as np 

x_time_df = pd.read_csv('dataset/x_time_train_test.csv')
y_df = pd.read_csv('dataset/y_train_test.csv')

# Convert grid representations to nested lists
x_time_df['env_input'] = x_time_df['env_input'].apply(lambda x: eval(x))

# Merge dataframes on Integer time step
merged_df = pd.merge(x_time_df, y_df, on='time_stamp')

# Convert input of nested list to nested list 
x_time = np.array(merged_df["env_input"].tolist())

# Load the trained model
model = CustomLSTMModel(input_size=input_size, lstm_hidden_size=lstm_hidden_size)
model.load_state_dict(torch.load('lstm.pth'))
model.eval()

# Prepare a sample input tensor (assuming it's similar to x_time_tensor)
sample_input = x_time_tensor[0].unsqueeze(0)  # Select the first sample and add a batch dimension

# Get the output predictions
with torch.no_grad():
    output = model(sample_input)

# Convert the output probabilities to predicted classes
_, predicted = torch.max(output, 2)

# Print the predicted classes
print(predicted)
