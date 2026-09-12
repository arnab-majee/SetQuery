import torch
import torch.nn as nn


class SAREncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 1024,
    ):
        super().__init__()

        self.encoder = nn.Sequential( 
            nn.Conv2d(input_channels,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.projection(features)