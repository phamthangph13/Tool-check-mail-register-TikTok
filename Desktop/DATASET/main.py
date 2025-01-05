import torch
import numpy as np  # Thêm dòng import NumPy
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train import Generator, test_model  # Giả sử bạn đã tạo file train.py chứa mã của Generator

# Hàm để hiển thị kết quả kiểm tra
def display_results(test_image_path, generated_image):
    # Đọc ảnh gốc
    original_image = Image.open(test_image_path).convert('RGB')
    
    # Chuyển đổi ảnh từ tensor về dạng có thể hiển thị
    generated_image = generated_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    original_image = original_image.resize((256, 256))
    original_image = np.array(original_image)

    # Vẽ ảnh gốc và ảnh quá khứ được sinh ra
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Modern Image")
    ax[0].axis('off')
    
    ax[1].imshow(generated_image)
    ax[1].set_title("Generated Historical Image")
    ax[1].axis('off')
    
    plt.show()

def main():
    # Đảm bảo thiết bị CUDA (GPU) hoặc CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tải mô hình đã huấn luyện
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('generator.pth', map_location=device))  # Tải mô hình đã huấn luyện

    # Đường dẫn đến ảnh thử nghiệm
    test_image_path = "test.jpg"
    
    # Kiểm tra mô hình với ảnh thử nghiệm
    historical_image = test_model(generator, test_image_path, device)
    
    # Hiển thị kết quả
    display_results(test_image_path, historical_image)

if __name__ == "__main__":
    main()
