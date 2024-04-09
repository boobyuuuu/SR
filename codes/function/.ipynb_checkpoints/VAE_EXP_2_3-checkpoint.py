# EXP_2_3: 更多的卷积层

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1) # 1
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim) # 1
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim) # 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x)) # 2
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8) # 3
        self.conv5 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, output_padding=0) # 3
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv3 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv1 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1, output_padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 8, 8) # 4
        x = F.relu(self.conv5(x)) # 4
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar