import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_mean_std(features: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = features.var(dim=(2, 3), unbiased=False, keepdim=True).add(eps).sqrt()
    return mean, std


def adaptive_instance_normalization(
    content_features: torch.Tensor,
    style_mean: torch.Tensor,
    style_std: torch.Tensor,
) -> torch.Tensor:
    content_mean, content_std = channel_mean_std(content_features)
    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean


class ScannerAdaINAutoencoder(nn.Module):
    def __init__(self, image_size: int = 256):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        image = self.decoder(features)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
