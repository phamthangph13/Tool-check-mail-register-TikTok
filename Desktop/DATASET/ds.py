import os
def count_image_files_in_folder(folder_path):
    try:
        # Các phần mở rộng tệp ảnh phổ biến
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        
        # Lấy tất cả các tệp tin trong thư mục
        files = os.listdir(folder_path)
        
        # Lọc ra các tệp ảnh
        image_files = [file for file in files if os.path.isfile(os.path.join(folder_path, file)) and 
                       any(file.lower().endswith(ext) for ext in image_extensions)]
        
        return len(image_files)
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Đường dẫn đến thư mục Past và Current
past_folder = "PAST"
current_folder = "CURRENT"

# Đếm số lượng tệp ảnh trong thư mục Past và Current
past_image_count = count_image_files_in_folder(past_folder)
current_image_count = count_image_files_in_folder(current_folder)

print(f"Số lượng tệp ảnh trong thư mục 'Past': {past_image_count}")
print(f"Số lượng tệp ảnh trong thư mục 'Current': {current_image_count}")