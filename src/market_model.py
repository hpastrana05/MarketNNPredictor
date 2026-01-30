import torch
import torch.nn as nn


class MarketLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # The LSTM layer: processes the sequence
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # The Linear layer: maps LSTM output to a single predicted price
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        # input_seq shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(input_seq)
        
        # We only care about the output of the VERY LAST time step in the sequence
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions


