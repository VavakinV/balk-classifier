import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 20

ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")

def rotated_rect_to_aabb(center_x, center_y, width, height, rotation_degrees):
    """Преобразование rotated rectangle в axis-aligned bounding box (AABB)"""
    rotation_rad = np.radians(rotation_degrees)
    vertices = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ])
    rot_mat = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])
    rotated_vertices = np.dot(vertices, rot_mat)
    rotated_vertices[:, 0] += center_x
    rotated_vertices[:, 1] += center_y
    x_min, y_min = np.min(rotated_vertices, axis=0)
    x_max, y_max = np.max(rotated_vertices, axis=0)
    return x_min, y_min, x_max, y_max

def parse_annotation(bbox_dict, original_width, original_height):
    """Извлечение координат AABB из аннотации"""
    cx = bbox_dict['x'] / 100.0 * original_width
    cy = bbox_dict['y'] / 100.0 * original_height
    width = bbox_dict['width'] / 100.0 * original_width
    height = bbox_dict['height'] / 100.0 * original_height
    rotation = bbox_dict['rotation']
    return rotated_rect_to_aabb(cx, cy, width, height, rotation)

def load_data(csv_path, img_base_dir):
    """Загрузка данных с проверкой путей"""
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        try:
            bbox_list = json.loads(row['code_bbox'].replace("'", '"'))
            bbox_dict = bbox_list[0]
            img_name = os.path.basename(row['image'])
            img_path = os.path.join(img_base_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                continue
                
            h, w = img.shape[:2]
            if h != bbox_dict['original_height'] or w != bbox_dict['original_width']:
                print(f"Warning: Size mismatch for {img_path}")
                continue
                
            x_min, y_min, x_max, y_max = parse_annotation(bbox_dict, w, h)
            data.append({
                'img_path': img_path,
                'orig_width': w,
                'orig_height': h,
                'bbox': [x_min, y_min, x_max, y_max]
            })
        except Exception as e:
            print(f"Error processing row {row}: {str(e)}")
    return data

def preprocess_image(img, target_size):
    """Предобработка изображения: resize и нормализация"""
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img


def create_model(input_shape):
    """Создание модели"""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(4, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse'
    )
    return model

def compute_iou(box1, box2):
    """Вычисление IoU для двух bounding box"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / (area1 + area2 - intersection_area)

# Загрузка данных
data = load_data(ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Подготовка данных для обучения
def data_generator(data, batch_size, target_size):
    while True:
        indices = np.random.randint(0, len(data), batch_size)
        batch_imgs = []
        batch_bboxes = []
        for idx in indices:
            sample = data[idx]
            img = cv2.imread(sample['img_path'])
            if img is None:
                continue
            # Предобработка изображения
            img_proc = preprocess_image(img, target_size)
            # Нормализация координат bbox
            x_min, y_min, x_max, y_max = sample['bbox']
            bbox_norm = [
                x_min / sample['orig_width'],
                y_min / sample['orig_height'],
                x_max / sample['orig_width'],
                y_max / sample['orig_height']
            ]
            batch_imgs.append(img_proc)
            batch_bboxes.append(bbox_norm)
        yield np.array(batch_imgs), np.array(batch_bboxes)

# Создание и обучение модели
model = create_model((TARGET_SIZE[1], TARGET_SIZE[0], 3))
train_gen = data_generator(train_data, BATCH_SIZE, TARGET_SIZE)
val_gen = data_generator(val_data, BATCH_SIZE, TARGET_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_data) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(val_data) // BATCH_SIZE
)

# Оценка на валидации
val_ious = []
for sample in val_data:
    img = cv2.imread(sample['img_path'])
    img_proc = preprocess_image(img, TARGET_SIZE)
    pred_bbox_norm = model.predict(np.array([img_proc]))[0]
    orig_h, orig_w = img.shape[:2]
    pred_bbox = [
        pred_bbox_norm[0] * orig_w,
        pred_bbox_norm[1] * orig_h,
        pred_bbox_norm[2] * orig_w,
        pred_bbox_norm[3] * orig_h
    ]
    iou = compute_iou(sample['bbox'], pred_bbox)
    val_ious.append(iou)

print(f"Mean IoU on validation: {np.mean(val_ious):.4f}")

# Тестирование на тестовых данных
test_dirs = [
    './dataset/test/altai',
    './dataset/test/begickaya',
    './dataset/test/promlit',
    './dataset/test/ruzhimmash',
    './dataset/test/tihvin'
]

test_images = []
for dir_path in test_dirs:
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(dir_path, filename))


def visualize_results(model, test_images, target_size, n=5):
    """Визуализация с корректным преобразованием координат"""
    for img_path in test_images[:n]:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        orig_h, orig_w = img.shape[:2]

        display_img = img.copy()
        
        img_proc = cv2.resize(img, target_size)
        img_proc = img_proc / 255.0

        pred_bbox_norm = model.predict(np.array([img_proc]))[0]
        
        x_min = int(pred_bbox_norm[0] * orig_w)
        y_min = int(pred_bbox_norm[1] * orig_h)
        x_max = int(pred_bbox_norm[2] * orig_w)
        y_max = int(pred_bbox_norm[3] * orig_h)

        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(os.path.basename(img_path))
        plt.show()

visualize_results(model, test_images, TARGET_SIZE)