import os
import sys
from pathlib import Path
import imgaug.augmenters as iaa
# 設定 PYTHONPATH
yolov5_path = 'D:/test3/yolov5'
sys.path.append(yolov5_path)

from models.experimental import attempt_load
from utils.general import check_dataset, increment_path
from utils.plots import plot_results
from utils.callbacks import Callbacks

# 設置資料夾路徑
train_images_folder = 'D:/test3/container_dataset/test'
train_annotations_folder = 'D:/test3/container_dataset/test/test_xml'
val_images_folder = 'D:/test3/container_dataset/val'
val_annotations_folder = 'D:/test3/container_dataset/val_xml'
test_images_folder = 'D:/test3/container_dataset/test'
test_annotations_folder = 'D:/test3/container_dataset/test_xml'

# YOLOv5 資料集目錄
dataset_dir = 'D:/test3/container_dataset'
output_dir = 'D:/test3/'
top_predictions_dir = os.path.join(output_dir, 'top_predictions')
best_model_path = os.path.join(output_dir, 'best.pt')

os.makedirs(os.path.join(dataset_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'val/labels'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'test/labels'), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(top_predictions_dir, exist_ok=True)

# 生成 data.yaml 文件
def create_data_yaml():
    yaml_content = f"""
train: {dataset_dir}/train/images
val: {dataset_dir}/val/images
test: {dataset_dir}/test/images

nc: 1  # number of classes
names: ['container']  # class names
"""
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

create_data_yaml()

# 修正標籤文件中的坐標值
def normalize_coordinates(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    corrected_lines = []
    for line in lines:
        line = line.strip()  # 確保每行是一個字符串
        parts = line.split()
        if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            corrected_lines.append(f"{cls} {x} {y} {w} {h}\n")
    with open(annotation_path, 'w') as file:
        file.writelines(corrected_lines)

def preprocess_annotations(annotations_folder):
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith(".txt"):
            annotation_path = os.path.join(annotations_folder, annotation_file)
            normalize_coordinates(annotation_path)

preprocess_annotations(os.path.join(dataset_dir, 'train/labels'))
preprocess_annotations(os.path.join(dataset_dir, 'val/labels'))
preprocess_annotations(os.path.join(dataset_dir, 'test/labels'))

# 資料增強
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Crop(percent=(0, 0.1)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Multiply((0.8, 1.2)),
    iaa.Affine(
        rotate=(-20, 20),
        shear=(-10, 10)
    )
])

def augment_image(image):
    return augmenters.augment_image(image)

# 訓練YOLOv5
def train_yolov5():
    data = os.path.join(dataset_dir, 'data.yaml')
    epochs = 100
    batch_size = 16
    img_size = 640
    weights = 'yolov5s.pt'
    project = output_dir
    name = 'exp'
    save_dir = increment_path(Path(project) / name, exist_ok=False)

    
    os.makedirs(save_dir, exist_ok=True)

    # early stopping
    patience = 10

    # 執行訓練
    from train import run
    run(data=data, epochs=epochs, batch_size=batch_size, imgsz=img_size, weights=weights, project=project, name=name, patience=patience, save_dir=save_dir)

    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        os.rename(best_model_path, os.path.join(output_dir, 'best.pt'))

    results = os.path.join(save_dir, 'results.txt')
    if os.path.exists(results):
        with open(results, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'F1' in line:
                    print(line.strip())

if __name__ == "__main__":
    train_yolov5()