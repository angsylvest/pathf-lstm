
import torch
import torch.nn as nn
import torch.optim as optim

# Define custom activation function to round the outputs to the nearest integer
def round_to_grid(x):
    return torch.round(x)

# Define custom LSTM model
class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size):
        super(CustomLSTMModel, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(lstm_hidden_size, 2)  # Assuming a 3x3 grid

    def forward(self, x):
        # Process input through LSTM
        lstm_out, _ = self.lstm(x)

        # Fully connected layer for final output
        output = self.fc(lstm_out[:, -1, :])

        return output