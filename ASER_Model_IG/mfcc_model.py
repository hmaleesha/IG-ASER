import torch
import torch.nn.functional as F
from torch import nn


#latest
class MFCC_CNNLSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # First CNN block
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mp = nn.MaxPool2d(2, 2)
        self.dr = nn.Dropout(0.5)

        # LSTM layer
        self.lstm_layer = nn.LSTM(128, 128, 1, bidirectional=True)  # Adjust input size based on CNN output shape

        # Fully connected layers
        self.linear1 = nn.Linear(128 * 2, 32)  # LSTM has 128 hidden units in both directions
        self.linear2 = nn.Linear(32, 4)        # Final output layer

    def forward(self, input_logits):
        # CNN layers
        cnn1_out = self.cnn1(input_logits)
        cnn1_out = nn.functional.relu(cnn1_out)
        cnn1_out = self.mp(cnn1_out)

        cnn2_out = self.cnn2(cnn1_out)
        cnn2_out = nn.functional.relu(cnn2_out)
        cnn2_out = self.mp(cnn2_out)

        cnn3_out = self.cnn3(cnn2_out)
        cnn3_out = nn.functional.relu(cnn3_out)
        cnn3_out = self.mp(cnn3_out)

        # Flatten the output for LSTM input
        fl = torch.flatten(cnn3_out, start_dim=1)

        # LSTM layer
        lstm_out, _ = self.lstm_layer(fl.unsqueeze(1))  # Add sequence dimension for LSTM
        lstm_out = lstm_out[:, -1, :]  # Get the last time step

        # Fully connected layers
        l1_out = self.linear1(lstm_out)
        dr_out = self.dr(l1_out)
        l2_out = self.linear2(dr_out)

        # Softmax for output
        output = nn.functional.softmax(l2_out, dim=1)

        # Return intermediate outputs and final output
        return cnn1_out, cnn2_out, cnn3_out, fl, lstm_out, l1_out, dr_out, l2_out, output
