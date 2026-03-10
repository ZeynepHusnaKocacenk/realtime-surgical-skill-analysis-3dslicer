import torch.nn as nn

# ================================
#  BiLSTM Model Definition
# ================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
