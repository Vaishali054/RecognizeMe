import torch
import torch.nn as nn
import torchvision.models as models

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Pre-trained feature extractor 
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer
        input_size = 512  

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # * 2 for bidirectional

    def forward(self, x):
        # Extract features from images
        batch_size = x.size(0)
        x = self.feature_extractor(x)

        # Reshape to (batch_size, seq_len=1, input_size)
        x = x.view(batch_size, 1, -1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
