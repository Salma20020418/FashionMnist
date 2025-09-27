import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional encoder that maps (B,1,28,28) -> (B, latent_dim)."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x7x7
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    """Convolutional decoder that maps (B, latent_dim) -> (B,1,28,28)."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 32x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 1x28x28
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x_hat = self.deconv(x)
        return x_hat


class Autoencoder(nn.Module):
    """Autoencoder combining the Encoder and Decoder."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick smoke test when running this file directly
    model = Autoencoder(latent_dim=64)
    print("Model created. Trainable parameters:", count_parameters(model))

    # random input batch (untrained network) - check shapes
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print("input shape:", x.shape)
    print("output shape:", out.shape)
    print("output range: [{:.4f}, {:.4f}]".format(out.min().item(), out.max().item()))
