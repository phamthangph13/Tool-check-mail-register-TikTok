import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial transformation network
        self.localization = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128 * 16 * 16)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class EnhancedGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        
        # Load pretrained ResNet for feature extraction
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        
        # Historical style encoder
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        # Spatial transformer for geometric changes
        self.spatial_transformer = SpatialTransformer()
        
        # Architecture transformer
        self.architecture_transformer = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Decoder with attention
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
        
        # Self-attention layers
        self.attention1 = SelfAttention(256)
        self.attention2 = SelfAttention(128)
        
    def forward(self, x, historical_style=None):
        # Extract deep features
        deep_features = self.feature_extractor(x)
        
        # Content encoding
        content = self.content_encoder(x)
        
        # Apply spatial transformation
        transformed = self.spatial_transformer(content)
        
        # Transform architecture
        architecture = self.architecture_transformer(transformed)
        
        # Decode with attention
        out = self.decoder[0:3](architecture)
        out = self.attention1(out)
        out = self.decoder[3:6](out)
        out = self.attention2(out)
        out = self.decoder[6:](out)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma*out + x

def enhanced_loss(fake_historical, historical, fake_features, real_features):
    # Content loss using deep features
    content_loss = F.mse_loss(fake_features, real_features)
    
    # Structural loss
    structural_loss = F.l1_loss(fake_historical, historical)
    
    # Perceptual loss using VGG features
    vgg = models.vgg16(pretrained=True).features.eval()
    fake_vgg = get_vgg_features(fake_historical, vgg)
    real_vgg = get_vgg_features(historical, vgg)
    perceptual_loss = sum(F.mse_loss(f, r) for f, r in zip(fake_vgg, real_vgg))
    
    # Style loss
    style_loss = sum(gram_loss(f, r) for f, r in zip(fake_vgg, real_vgg))
    
    return content_loss + structural_loss + 0.1 * perceptual_loss + 0.5 * style_loss

def get_vgg_features(x, model):
    features = []
    for i, layer in enumerate(model):
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            features.append(x)
    return features

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h*w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

def gram_loss(fake, real):
    return F.mse_loss(gram_matrix(fake), gram_matrix(real))
