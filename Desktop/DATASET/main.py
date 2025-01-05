import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import datetime

def train_historical_model(modern_dir, historical_dir, epochs=100, batch_size=4, save_dir="checkpoints"):
    # Tạo thư mục lưu model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Chuẩn bị transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = HistoricalDataset(modern_dir, historical_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Khởi tạo models
    generator = EnhancedGenerator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Schedulers
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.5)
    
    # Khởi tạo VGG cho perceptual loss
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    
    # Training loop
    best_loss = float('inf')
    start_time = datetime.datetime.now()
    
    try:
        for epoch in range(epochs):
            generator.train()
            total_g_loss = 0
            total_batches = 0
            
            for i, (modern, historical) in enumerate(dataloader):
                modern, historical = modern.to(device), historical.to(device)
                batch_size = modern.size(0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                real_output = discriminator(historical)
                fake_historical = generator(modern)
                fake_output = discriminator(fake_historical.detach())
                
                d_loss = (F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) + 
                         F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))) / 2
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                fake_output = discriminator(fake_historical)
                
                # Calculate losses
                fake_features = generator.feature_extractor(fake_historical)
                real_features = generator.feature_extractor(historical)
                
                g_loss = enhanced_loss(fake_historical, historical, fake_features, real_features) + \
                        F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
                
                g_loss.backward()
                g_optimizer.step()
                
                total_g_loss += g_loss.item()
                total_batches += 1
                
                # Print progress
                if i % 10 == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] '
                          f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} '
                          f'Time: {elapsed_time}')
            
            # Update schedulers
            g_scheduler.step()
            d_scheduler.step()
            
            # Save checkpoint after each epoch
            avg_loss = total_g_loss / total_batches
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(generator.state_dict(), os.path.join(save_dir, 'best_generator.pth'))
                
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        torch.save(generator.state_dict(), os.path.join(save_dir, 'interrupted_generator.pth'))
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
        
    return generator

def restore_historical_image(model_path, input_image_path, output_path):
    """
    Sử dụng model đã train để phục dựng ảnh lịch sử
    """
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    generator = EnhancedGenerator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Chuẩn bị transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load và xử lý ảnh đầu vào
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Generate ảnh lịch sử
    with torch.no_grad():
        historical_image = generator(input_tensor)
        
    # Chuyển tensor thành ảnh và lưu
    historical_image = (historical_image + 1) / 2  # denormalize
    historical_image = historical_image.squeeze(0).cpu()
    historical_image = transforms.ToPILImage()(historical_image)
    historical_image.save(output_path)
    print(f"Restored image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Đường dẫn thư mục chứa ảnh train
    modern_dir = "dataset/modern"  # Thư mục chứa ảnh hiện đại
    historical_dir = "dataset/historical"  # Thư mục chứa ảnh lịch sử tương ứng
    
    # Training
    print("Starting training...")
    generator = train_historical_model(modern_dir, historical_dir, epochs=100)
    print("Training completed!")
    
    # Testing
    print("Testing model...")
    model_path = "checkpoints/best_generator.pth"
    input_image = "test/modern_image.jpg"
    output_image = "test/historical_restored.jpg"
    
    restore_historical_image(model_path, input_image, output_image)
    print("Testing completed!")
