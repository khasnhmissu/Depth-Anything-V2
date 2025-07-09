import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' 
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 80 # 20 for indoor model, 80 for outdoor model


print(f"Using device: {DEVICE}")
print(f"Loading model with encoder: {encoder}")

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# Đảm bảo đường dẫn tới file checkpoint là chính xác
model_path = 'depth_anything_v2_metric_vkitti_vitl.pth'
if not os.path.exists(model_path):
    print(f"Error: Checkpoint file not found at {model_path}")
    print("Please download the checkpoint file and place it in the 'checkpoints' directory.")
else:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Model loaded successfully.")

    # --- Phần xử lý và hiển thị ảnh --
    image_path = "E:\\dataset_citiscape\\clear\\train\\images\\aachen_000014_000019_leftImg8bit_aug_0.png"

    # Kiểm tra xem tệp có tồn tại không
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
    else:
        # 1. Đọc ảnh bằng OpenCV
        raw_img = cv2.imread(image_path)

        # 2. Tạo depth map
        print("Inferring depth...")
        depth = model.infer_image(raw_img) # HxW raw depth map in numpy
        print("Inference complete.")
        print(depth.size)
        print(depth) # Bạn có thể bỏ comment dòng này nếu vẫn muốn xem ma trận numpy

        # 3. Hiển thị ảnh gốc và depth map
        
        # Chuyển đổi ảnh gốc từ BGR (OpenCV) sang RGB (Matplotlib) để màu sắc hiển thị đúng
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Tạo một figure có 2 subplot (1 hàng, 2 cột)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Hiển thị ảnh gốc
        axes[0].imshow(raw_img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off') # Ẩn các trục tọa độ
        
        # Hiển thị depth map
        # Sử dụng colormap 'viridis' hoặc 'magma' để dễ nhìn hơn
        axes[1].imshow(depth, cmap='magma')
        axes[1].set_title('Depth Map')
        axes[1].axis('off') # Ẩn các trục tọa độ
        
        # Hiển thị cửa sổ plot
        plt.show()