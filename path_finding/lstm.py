'''
Generic LSTM + usage example (would be broken down into multiple files eventually)

'''
import torch
import torch.nn as nn
import torch.optim as optim

class CustomLSTMModel(nn.Module):
    def __init__(self, num_time_features, num_context_features, lstm_hidden_size):
        super(CustomLSTMModel, self).__init__()

        # LSTM layers for time-based features
        self.lstm_time = nn.LSTM(input_size=num_time_features, hidden_size=lstm_hidden_size)

        # LSTM layers for contextual features
        self.lstm_context = nn.LSTM(input_size=num_context_features, hidden_size=lstm_hidden_size)

        # Fully connected layer for combining outputs
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)  # Multiply by 2 as we concatenate outputs

    def forward(self, x_time, x_context):
        # Process time-based features through LSTM
        lstm_time_out, _ = self.lstm_time(x_time)

        # Process contextual features through LSTM
        lstm_context_out, _ = self.lstm_context(x_context)

        # Concatenate the outputs along the last dimension (time steps)
        merged = torch.cat((lstm_time_out[:, -1, :], lstm_context_out[:, -1, :]), dim=1)

        # Fully connected layer for final output
        output = self.fc(merged)

        return output
