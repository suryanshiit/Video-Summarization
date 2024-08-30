import torch.nn as nn

class vsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(vsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, 1)  # bidirectional doubles hidden_size

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm)
        return out

# Example usage
model = vsLSTM(input_size=1024, hidden_size=512, num_layers=2)

import torch.nn as nn

class vsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(vsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, 1)  # bidirectional doubles hidden_size

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm)
        return out

# Example usage
model = vsLSTM(input_size=1024, hidden_size=512, num_layers=2)

from sklearn.metrics import f1_score

def evaluate(predictions, ground_truth):
    predictions = (predictions > 0.5).float()
    f1 = f1_score(ground_truth, predictions, average='weighted')
    return f1
