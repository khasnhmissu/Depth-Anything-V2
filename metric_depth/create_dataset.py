import cv2
import torch, gc
import numpy as np
import os
import shutil
from tqdm import tqdm

# Import model từ thư viện depth_anything_v2
# Đảm bảo bạn đã cài đặt thư viện này và có thể import
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Lỗi: Không tìm thấy thư viện 'depth_anything_v2'.")
    print("Vui lòng đảm bảo bạn đã cài đặt và cấu trúc thư mục đúng.")
    exit()

# ==============================================================================
# ================================= CẤU HÌNH ===================================
# ==============================================================================

# 1. ĐƯỜNG DẪN
# !!! THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN !!!
INPUT_ROOT_DIR = './WIDER_face'
OUTPUT_ROOT_DIR = './WIDER_face_foggy'
CHECKPOINT_PATH = './depth_anything_v2_metric_vkitti_vitl.pth' # Đặt file này cùng thư mục

# 2. CẤU HÌNH MODEL DEPTH
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
ENCODER = 'vitl'
MAX_DEPTH = 80  # 80m cho mô hình ngoài trời (vkitti)

# 3. CẤU HÌNH TẠO SƯƠNG MÙ
# Chọn beta ngẫu nhiên trong khoảng này để tạo độ đậm nhạt sương mù khác nhau
FOG_LEVELS = {
    'foggy_medium': {
        'name': 'Sương mù Vừa',
        'beta_min': 0.015,
        'beta_max': 0.025
    },

}

# ==============================================================================
# ======================== CÁC HÀM TIỆN ÍCH TẠO SƯƠNG MÙ =======================
# ==============================================================================

def estimate_atmosphere_light(image, dark_channel_window_size=15):
    """
    Ước lượng ánh sáng khí quyển (màu của sương mù) từ ảnh.
    image: ảnh RGB float trong khoảng [0, 1].
    """
    # 1. Tính dark channel
    min_rgb = np.min(image, axis=2)
    kernel = np.ones((dark_channel_window_size, dark_channel_window_size), np.uint8)
    dark = cv2.erode(min_rgb, kernel)

    # 2. Tìm 0.1% pixel sáng nhất trong dark channel
    num_pixels = int(0.001 * dark.size)
    flat_dark = dark.flatten()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    
    # 3. Lấy giá trị màu trung bình của các pixel tương ứng trong ảnh gốc
    # để làm ánh sáng khí quyển A
    rows, cols = np.unravel_index(indices, dark.shape)
    atmosphere_light = np.mean(image[rows, cols], axis=0)
    
    return atmosphere_light

def generate_foggy_image(image_rgb_float, depth_map, beta, atmosphere_light):
    """
    Tạo ảnh sương mù từ ảnh gốc, depth map và các tham số.
    image_rgb_float: ảnh RGB float trong khoảng [0, 1].
    depth_map: bản đồ độ sâu theo mét.
    beta: hệ số tán xạ (độ đậm của sương mù).
    atmosphere_light: vector màu của ánh sáng khí quyển [R, G, B].
    """
    # Mô hình vật lý: I_foggy(x) = I_clear(x) * t(x) + A * (1 - t(x))
    # trong đó t(x) = exp(-beta * d(x)) là hệ số truyền qua (transmittance)
    depth_smooth = cv2.GaussianBlur(depth_map, (5, 5), 0)
    # 1. Tính hệ số truyền qua (transmittance)
    transmittance = np.exp(-beta * depth_smooth)
    
    # 2. Mở rộng chiều để tính toán (broadcasting)
    # t từ (H, W) -> (H, W, 1)
    # A từ (3,) -> (1, 1, 3)
    t = np.expand_dims(transmittance, axis=-1)
    A = np.expand_dims(np.expand_dims(atmosphere_light, axis=0), axis=0)
    
    # 3. Áp dụng công thức
    foggy_image_float = image_rgb_float * t + A * (1 - t)
    
    # 4. Chuyển đổi về định dạng ảnh 8-bit [0, 255]
    foggy_image_uint8 = np.clip(foggy_image_float * 255, 0, 255).astype(np.uint8)
    
    return foggy_image_uint8

# ==============================================================================
# ================================== HÀM MAIN ==================================
# ==============================================================================

def main():
    # ----- 1. Load model Depth Anything V2 -----
    print(f"Sử dụng thiết bị: {DEVICE}")
    print(f"Đang tải model Depth Anything V2 (encoder: {ENCODER}, max_depth: {MAX_DEPTH}m)...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Lỗi: Không tìm thấy file checkpoint tại '{CHECKPOINT_PATH}'")
        return
        
    depth_model = DepthAnythingV2(**{**MODEL_CONFIGS[ENCODER], 'max_depth': MAX_DEPTH})
    depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval().half()  # Chuyển sang half precision để tiết kiệm VRAM
    print("Tải model thành công.")
    
    # ----- 2. Lặp qua từng mức độ sương mù đã định nghĩa -----
    for level_key, level_config in FOG_LEVELS.items():
        
        current_output_root = os.path.join(OUTPUT_ROOT_DIR, level_key)
        level_name = level_config['name']
        beta_min = level_config['beta_min']
        beta_max = level_config['beta_max']
        
        print(f"\n\n{'='*20} BẮT ĐẦU XỬ LÝ MỨC ĐỘ: {level_name.upper()} {'='*20}")
        print(f"Thư mục đầu ra: {current_output_root}")
        print(f"Khoảng beta: [{beta_min}, {beta_max}]")
        
        # ----- 3. Bắt đầu xử lý các thư mục con (train, val, test) cho mức độ hiện tại -----
        sub_dirs = [d for d in os.listdir(INPUT_ROOT_DIR) if os.path.isdir(os.path.join(INPUT_ROOT_DIR, d))]
        
        for sub_dir in sub_dirs:
            input_image_dir = os.path.join(INPUT_ROOT_DIR, sub_dir, 'images')
            input_label_dir = os.path.join(INPUT_ROOT_DIR, sub_dir, 'labels')
            
            output_image_dir = os.path.join(current_output_root, sub_dir, 'images')
            output_label_dir = os.path.join(current_output_root, sub_dir, 'labels')
            
            print(f"\n---------- Đang xử lý thư mục con: {sub_dir} ----------")
            
            # Tạo cấu trúc thư mục đầu ra
            print(f"Tạo thư mục: {output_image_dir}")
            os.makedirs(output_image_dir, exist_ok=True)
            
            # Sao chép thư mục labels
            if os.path.exists(input_label_dir):
                print(f"Sao chép labels từ '{input_label_dir}' sang '{output_label_dir}'")
                if os.path.exists(output_label_dir):
                    shutil.rmtree(output_label_dir) # Xóa thư mục cũ nếu có để tránh lỗi
                shutil.copytree(input_label_dir, output_label_dir)
            
            # Lấy danh sách ảnh cần xử lý
            image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Sử dụng tqdm để tạo thanh tiến trình
            for image_name in tqdm(image_files, desc=f"Tạo sương mù '{level_name}' cho '{sub_dir}'"):
                image_path = os.path.join(input_image_dir, image_name)
                output_path = os.path.join(output_image_dir, image_name)
                
                # Đọc ảnh và chuyển sang RGB
                raw_img_bgr = cv2.imread(image_path)
                if raw_img_bgr is None:
                    print(f"\nCảnh báo: Bỏ qua file không đọc được: {image_path}")
                    continue
                raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
                raw_img_rgb = raw_img_rgb.half().to(DEVICE)
                # 1. Tạo depth map
                depth_map = depth_model.infer_image(raw_img_rgb)
                torch.cuda.empty_cache()
                gc.collect()
                # 2. Ước lượng ánh sáng khí quyển
                image_rgb_float = raw_img_rgb / 255.0
                atmosphere_light = estimate_atmosphere_light(image_rgb_float)
                
                # 3. Chọn ngẫu nhiên hệ số beta TRONG KHOẢNG CỦA MỨC ĐỘ HIỆN TẠI
                beta = np.random.uniform(beta_min, beta_max)
                
                # 4. Tạo ảnh sương mù
                foggy_image_rgb = generate_foggy_image(image_rgb_float, depth_map, beta, atmosphere_light)
                
                # 5. Chuyển lại BGR để lưu bằng OpenCV và lưu ảnh
                foggy_image_bgr = cv2.cvtColor(foggy_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, foggy_image_bgr)
                
    print("\n=====================================================================")
    print("✅ Hoàn thành! Đã tạo xong tất cả các bộ dữ liệu sương mù.")
    print(f"Dữ liệu được lưu tại thư mục gốc: {OUTPUT_ROOT_DIR}")

if __name__ == '__main__':
    main()