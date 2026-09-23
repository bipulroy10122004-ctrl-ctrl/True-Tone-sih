"""
True Tone SIH - Audio Deepfake & Anti-Spoofing Model
Architecture: 1D CNN Front-End -> Bidirectional LSTM -> Self-Attention -> Classifier Head
Optimized for low-latency inference and 4GB VRAM GPU footprint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Computes temporal attention weights to emphasize critical acoustic cues
    (e.g., vocoder harmonic glitches, unnatural phoneme transitions).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len=100, hidden_dim=256)
        energy = self.projection(lstm_output)         # (batch, seq_len, 1)
        weights = F.softmax(energy.squeeze(-1), dim=-1) # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1) # (batch, hidden_dim)
        return context, weights

class AudioSpoofDetector(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128):
        super().__init__()
        # 1D Convolution over frequency channels across time
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # Bidirectional Recurrent Layer for temporal consistency
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention Pooling Layer
        self.attention = SelfAttention(hidden_dim * 2)

        # Classification Head (outputs single scalar logit)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Positive = Spoof (High Risk), Negative = Bonafide (Safe)
        )

    def forward(self, x):
        # Gracefully handle 2D inputs: (time_steps, features) -> (1, time_steps, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Handle NaN / Inf inputs gracefully
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle missing feature channels (e.g. fewer than 60 channels provided)
        if x.size(-1) < 60:
            pad_feats = 60 - x.size(-1)
            x = F.pad(x, (0, pad_feats), mode='constant', value=0.0)
        elif x.size(-1) > 60:
            x = x[..., :60]

        # Handle missing time steps (ensure at least 4 frames for 2 MaxPool1d operations)
        if x.size(1) < 4:
            x = F.pad(x, (0, 0, 0, 4 - x.size(1)), mode='constant', value=0.0)

        # Input x shape: (batch_size, time_steps, features=60)
        x = x.permute(0, 2, 1)          # Conv1D expects (batch, channels=60, length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)                # (batch, 64, length/2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                # (batch, 128, 100)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)          # LSTM expects (batch, seq_len=100, features=128)
        lstm_out, _ = self.lstm(x)      # (batch, 100, hidden_dim*2 = 256)

        context, attn_weights = self.attention(lstm_out) # (batch, 256)
        logits = self.classifier(context)                # (batch, 1)
        return logits.squeeze(-1)                        # (batch,)
