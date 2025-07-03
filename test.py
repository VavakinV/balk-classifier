import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import cv2
import os
import csv
from main import rotated_rect_to_aabb, TRAIN_IMAGES_PATH, ANNOTATIONS_PATH

def load_annotations(annotations_path):
    annotations = []
    with open(annotations_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bbox_data = json.loads(row['code_bbox'].replace("'", '"'))
            if not bbox_data:
                continue
            
            bbox = bbox_data[0]  # Берем первый bounding box
            annotations.append({
                'image': os.path.basename(row['image']),
                'bbox': {
                    'x': float(bbox['x']),
                    'y': float(bbox['y']),
                    'width': float(bbox['width']),
                    'height': float(bbox['height']),
                    'rotation': float(bbox['rotation']),
                    'original_width': int(bbox['original_width']),
                    'original_height': int(bbox['original_height'])
                }
            })
    return annotations

# Визуализация
def visualize_aabb(image_path, bbox_data):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_height, img_width = img.shape[:2]

    # Получаем параметры из разметки
    bbox = bbox_data['bbox']
    width = bbox['width'] / 100 * img_width
    height = bbox['height'] / 100 * img_height
    cx = bbox['x'] / 100 * img_width + width/2
    cy = bbox['y'] / 100 * img_height + height/2
    rotation = bbox['rotation']
    
    left = cx - width/2
    top = cy - height/2

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    rect = Rectangle((left, top), width, height, 
                    angle=rotation, rotation_point=(left, top),
                    fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    
    x_min, y_min, x_max, y_max = rotated_rect_to_aabb(cx, cy, width, height, rotation)

    # Рисуем AABB
    aabb_width = x_max - x_min
    aabb_height = y_max - y_min
    rect_aabb = Rectangle((x_min, y_min), aabb_width, aabb_height,
                         fill=False, color='green', linewidth=2)
    ax.add_patch(rect_aabb)
    
    # Настройки отображения
    plt.title(f"Original (red) vs AABB (green)\nRotation: {rotation}°")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Основной код
annotations = load_annotations(ANNOTATIONS_PATH)
first_annotation = annotations[30]
image_path = os.path.join(TRAIN_IMAGES_PATH, first_annotation['image'])
visualize_aabb(image_path, first_annotation)