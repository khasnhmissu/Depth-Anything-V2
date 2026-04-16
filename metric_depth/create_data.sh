set -e

PYTHON_ENV_NAME="venv"
DATASET_GDRIVE_ID="1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b"
MODEL_GDRIVE_ID="1evX7_WtdNEQ-26TNDQADC44xWXemAxNE"
DATASET_ZIP_NAME="source_real.zip"
DATASET_DIR_NAME="source_real"

MODEL_FILENAME="depth_anything_v2_metric_vkitti_vitl.pth"

echo "apt-get update..."
sudo apt-get update
echo "Cài đặt python3-venv và unzip..."
sudo apt-get install -y python3-venv unzip

if [ ! -d "$PYTHON_ENV_NAME" ]; then
    echo "🐍 Tạo môi trường ảo Python tên là '$PYTHON_ENV_NAME'..."
    python3 -m venv $PYTHON_ENV_NAME
else
    echo "🐍 Môi trường ảo '$PYTHON_ENV_NAME' đã tồn tại."
fi

echo "Kích hoạt môi trường ảo..."
source $PYTHON_ENV_NAME/bin/activate



echo "📦 Cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt


echo "🧠 Kiểm tra và tải model checkpoint..."

if [ -f "$MODEL_FILENAME" ]; then
    echo "Model checkpoint '$MODEL_FILENAME' đã tồn tại, bỏ qua bước tải."
else
    echo "Tải model checkpoint từ Google Drive (ID: $MODEL_GDRIVE_ID)..."
    # Sử dụng gdown để tải file và lưu với tên chính xác
    gdown -O "$MODEL_FILENAME" "$MODEL_GDRIVE_ID"
    echo "✅ Đã tải xong model checkpoint."
fi

echo "💾 Kiểm tra và tải dataset..."

if [ -d "$DATASET_DIR_NAME" ]; then
    echo "Thư mục dataset '$DATASET_DIR_NAME' đã tồn tại, bỏ qua bước tải và giải nén."
else
    if [ -f "$DATASET_ZIP_NAME" ]; then
        echo "File '$DATASET_ZIP_NAME' đã tồn tại, sẽ tiến hành giải nén."
    else
        echo "Tải dataset từ Google Drive (ID: $DATASET_GDRIVE_ID)..."
        gdown -O $DATASET_ZIP_NAME $DATASET_GDRIVE_ID
    fi

    echo "Giải nén dataset..."
    unzip -q $DATASET_ZIP_NAME
    echo "Đã giải nén xong vào thư mục '$DATASET_DIR_NAME'."
fi

echo "🔥 Bắt đầu tạo dataset sương mù..."
python create_dataset.py

echo "✅ Hoàn tất! Dataset sương mù đã được tạo trong thư mục foggy_source_real/"
zip -r foggy_source_real.zip foggy_source_real/
echo "Dataset đã được nén thành foggy_source_real.zip"