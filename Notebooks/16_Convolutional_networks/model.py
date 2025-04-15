import torch
import torch.nn as nn

class CNN_Segmenter(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  #   [B, 1, 128, 128] → [B, 16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → [B, 16, 64, 64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → [B, 32, 32, 32]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # → [B, 16, 64, 64]
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),   # → [B, 8, 128, 128]
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=1)                        # → [B, 1, 128, 128] (logits)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # raw logits
