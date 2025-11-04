import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128*7*7, latent_dim)
        self.fc_logvar = nn.Linear(128*7*7, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder
class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 128, 7, 7)
        x_hat = self.deconv(x)
        return x_hat

# Variational Autoencoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = VariationalDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
