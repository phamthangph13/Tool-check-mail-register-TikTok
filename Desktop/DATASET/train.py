import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MemoryModule(nn.Module):
    def __init__(self, memory_size=1000, feature_dim=512):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        
        # Enhanced projection layers
        self.query_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        self.memory_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, query):
        batch_size = query.size(0)
        h, w = query.size(2), query.size(3)
        
        # Project query features
        query_flat = self.query_proj(query)
        query_flat = query_flat.view(batch_size, self.feature_dim, -1)
        query_flat = query_flat.permute(0, 2, 1)
        
        # Project memory features
        memory_proj = self.memory_proj(self.memory)
        
        # Calculate attention with improved scaling
        scaling_factor = torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float32))
        attention = torch.matmul(query_flat, memory_proj.t()) / scaling_factor
        attention = F.softmax(attention, dim=-1)
        
        # Combine with memory
        output = torch.matmul(attention, memory_proj)
        output = output.permute(0, 2, 1).view(batch_size, self.feature_dim, h, w)
        
        return output

class Generator(nn.Module):
    def __init__(self, latent_dim=100, memory_size=1000):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Enhanced encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Enhanced memory module
        self.memory = MemoryModule(memory_size=memory_size, feature_dim=512)
        
        # Decoder with skip connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Final refinement layers
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        # Edge enhancement layer
        self.edge_detect = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        # Add 1x1 conv layers to adjust the channels of e3, e2, and e1
        self.conv1x1_e3 = nn.Conv2d(256, 512, 1, 1, 0)  # For matching channels with d1
        self.conv1x1_e2 = nn.Conv2d(128, 256, 1, 1, 0)  # For matching channels with d2
        self.conv1x1_e1 = nn.Conv2d(64, 128, 1, 1, 0)   # For matching channels with d3
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder[0:2](x)
        e2 = self.encoder[2:5](e1)
        e3 = self.encoder[5:8](e2)
        features = self.encoder[8:](e3)
        
        # Memory enhancement
        memory_features = self.memory(features)
        enhanced_features = features + 0.3 * memory_features
        
        # Decoder with skip connections
        d1 = self.decoder_block1(enhanced_features)
        
        # Apply the 1x1 conv to e3 before adding to d1
        e3_adjusted = self.conv1x1_e3(e3)
        d2 = self.decoder_block2(d1 + 0.1 * e3_adjusted)
        
        # Apply the 1x1 conv to e2 before adding to d2
        e2_adjusted = self.conv1x1_e2(e2)
        d3 = self.decoder_block3(d2 + 0.1 * e2_adjusted)
        
        # Apply the 1x1 conv to e1 before adding to d3
        e1_adjusted = self.conv1x1_e1(e1)
        d4 = self.decoder_block4(d3 + 0.1 * e1_adjusted)
        
        # Final refinement
        output = self.final(d4)
        
        # Edge enhancement
        edge_enhanced = self.edge_detect(output)
        final_output = output + 0.1 * edge_enhanced
        
        return final_output

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Initial layer
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # Intermediate layers with increased channels
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Additional layer for better feature extraction
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.model(x)

class HistoricalDataset(Dataset):
    def __init__(self, modern_dir, historical_dir, transform=None):
        self.modern_paths = [os.path.join(modern_dir, f) for f in os.listdir(modern_dir)]
        self.historical_paths = [os.path.join(historical_dir, f) for f in os.listdir(historical_dir)]
        self.transform = transform
        
    def __len__(self):
        return len(self.modern_paths)
        
    def __getitem__(self, idx):
        modern_img = Image.open(self.modern_paths[idx]).convert('RGB')
        historical_img = Image.open(self.historical_paths[idx]).convert('RGB')
        
        if self.transform:
            modern_img = self.transform(modern_img)
            historical_img = self.transform(historical_img)
            
        return modern_img, historical_img

def modified_loss(fake_historical, historical):
    # L1 loss for overall structure
    l1_loss = F.l1_loss(fake_historical, historical)
    
    # Edge loss for details
    edge_kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(fake_historical.device)
    edge_kernel = edge_kernel.repeat(3, 1, 1, 1)
    
    fake_edges = F.conv2d(fake_historical, edge_kernel, padding=1, groups=3)
    real_edges = F.conv2d(historical, edge_kernel, padding=1, groups=3)
    edge_loss = F.mse_loss(fake_edges, real_edges)
    
    # Combined loss
    return l1_loss + 0.1 * edge_loss

def train_model(modern_dir, historical_dir, num_epochs=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = HistoricalDataset(modern_dir, historical_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  # Batch size nhỏ hơn
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers with adjusted learning rates
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = modified_loss
    
    # Early stopping setup
    best_g_loss = float('inf')
    patience = 15  # Number of epochs to wait for improvement
    counter = 0
    
    try:
        for epoch in range(num_epochs):
            for i, (modern, historical) in enumerate(dataloader):
                batch_size = modern.size(0)
                modern, historical = modern.to(device), historical.to(device)
                
                # Labels for adversarial training
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                real_output = discriminator(historical)
                d_loss_real = adversarial_loss(real_output, real_labels)
                
                fake_historical = generator(modern)
                fake_output = discriminator(fake_historical.detach())
                d_loss_fake = adversarial_loss(fake_output, fake_labels)
                
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                fake_output = discriminator(fake_historical)
                g_loss_gan = adversarial_loss(fake_output, real_labels)
                g_loss_l1 = reconstruction_loss(fake_historical, historical)
                
                # Adjusted loss weights
                g_loss = g_loss_gan + 150 * g_loss_l1
                g_loss.backward()
                g_optimizer.step()
                
                # Print progress
                if i % 10 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}], '
                          f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                    
                # Save checkpoints
                if i % 100 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{i}.pth')
                    torch.save({
                        'epoch': epoch,
                        'step': i,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'g_optimizer_state_dict': g_optimizer.state_dict(),
                        'd_optimizer_state_dict': d_optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # Early stopping check
            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
    return generator, discriminator


def test_model(generator, test_image_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_image = Image.open(test_image_path).convert('RGB')
    test_tensor = transform(test_image).unsqueeze(0).to(device)
    
    generator.eval()
    with torch.no_grad():
        historical_image = generator(test_tensor)
    
    # Denormalize the output
    historical_image = (historical_image + 1) / 2
    return historical_image

if __name__ == "__main__":
    modern_dir = "CURRENT"
    historical_dir = "PAST"
    
    print("Starting training...")
    generator, discriminator = train_model(modern_dir, historical_dir, num_epochs=50)
    
    print("Saving model...")
    torch.save(generator.state_dict(), 'generator.pth')
    
    print("Testing model...")
    test_image_path = "test.jpg"
    historical_image = test_model(generator, test_image_path)
    print("Done!")