import torch
import torch.nn as nn
import torch.optim as optim

def loss_function(x, u, x_recon, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) -logvar.exp())
    sparity_loss = torch.norm(u, p=1)
    return recon_loss + kld_loss + sparity_loss

class SparseTransitionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers=2, l2_weight=0.001):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Sparse Transition Module
        self.transition_fn = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )


    def encode(self, x):
        hidden = self.encoder(x)
        mu, logvar = self.fc_mu(hidden), self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z) 
        return x_recon

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size*703, 1, 41))
        mu, logvar = self.encode(x)
        mu = mu.reshape((batch_size, 703, -1))
        logvar = logvar.reshape((batch_size, 703, -1))
        z = self.reparameterize(mu, logvar)
        # Sparse Transition
        u, (h, c) = self.transition_fn(z)  # Apply transition to all but last z
        # Reconstruction
        x_recon = self.decode(u)
        x_recon = x_recon.reshape((batch_size*703, 1, 41))
        # Loss calculation
        loss = loss_function(x, u, x_recon, mu, logvar)
        return x_recon, z, u, loss