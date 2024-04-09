# EXP3_1: 引入L2正则化和Dropout正则化

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.l2_reg = nn.Linear(1024 * 8 * 8, 1)  # L2正则化

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)  # 在第一个卷积层后应用 Dropout
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # 在第二个卷积层后应用 Dropout
        x = F.relu(self.conv3(x))
        x = self.dropout(x)  # 在第三个卷积层后应用 Dropout
        x = F.relu(self.conv4(x))
        x = self.dropout(x)  # 在第四个卷积层后应用 Dropout
        x = F.relu(self.conv5(x))
        x = self.dropout(x)  # 在第五个卷积层后应用 Dropout
        x = x.view(x.size(0), -1)
        # L2 正则化
        l2_penalty = torch.sum(torch.pow(self.l2_reg.weight, 2))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # 添加 L2 正则化项到损失函数
        l2_loss = 0.001 * l2_penalty
        return mu, logvar, l2_loss

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)
        self.conv5 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv3 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv1 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1, output_padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 8, 8)
        x = F.relu(self.conv5(x))
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
        mu, logvar, l2_loss = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, l2_loss
