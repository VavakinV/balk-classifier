import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 20

ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")

def rotated_rect_to_aabb(center_x, center_y, width, height, rotation_degrees):
    """Преобразование rotated rectangle в axis-aligned bounding box (AABB)"""
    # Проверка на нулевой угол
    if abs(rotation_degrees) < 1e-6:
        return (
            center_x - width/2,
            center_y - height/2,
            center_x + width/2,
            center_y + height/2
        )
    
    rotation_rad = np.radians(rotation_degrees)

    corners = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ])
    rot_mat = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])
    rotated_corners = np.dot(corners, rot_mat)

    # Смещение центра
    offset_x = (width/2 * np.cos(rotation_rad) - height/2 * np.sin(rotation_rad)) - width/2
    offset_y = (width/2 * np.sin(rotation_rad) + height/2 * np.cos(rotation_rad)) - height/2
    true_center_x = center_x + offset_x
    true_center_y = center_y + offset_y

    # Перерасчёт координат углов
    rotated_corners[:, 0] += true_center_x
    rotated_corners[:, 1] += true_center_y

    x_min, y_min = np.min(rotated_corners, axis=0)
    x_max, y_max = np.max(rotated_corners, axis=0)
    return x_min, y_min, x_max, y_max

def parse_annotation(bbox_dict, original_width, original_height):
    """Извлечение координат AABB из аннотации"""
    width = bbox_dict['width'] / 100.0 * original_width
    height = bbox_dict['height'] / 100.0 * original_height
    cx = bbox_dict['x'] / 100.0 * original_width + width/2
    cy = bbox_dict['y'] / 100.0 * original_height + height/2
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


def iou_metric(y_true, y_pred):
    """
    Метрика IoU для модели Keras.
    y_true и y_pred - тензоры формы (batch_size, 4) с координатами [x_min, y_min, x_max, y_max]
    """
    # Разделяем координаты
    true_x1, true_y1, true_x2, true_y2 = tf.unstack(y_true, 4, axis=-1)
    pred_x1, pred_y1, pred_x2, pred_y2 = tf.unstack(y_pred, 4, axis=-1)
    
    # Вычисляем координаты пересечения
    intersect_x1 = tf.maximum(true_x1, pred_x1)
    intersect_y1 = tf.maximum(true_y1, pred_y1)
    intersect_x2 = tf.minimum(true_x2, pred_x2)
    intersect_y2 = tf.minimum(true_y2, pred_y2)
    
    # Площадь пересечения
    intersect_area = tf.maximum(0.0, intersect_x2 - intersect_x1) * tf.maximum(0.0, intersect_y2 - intersect_y1)
    
    # Площади каждого прямоугольника
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    
    # Объединение и IoU
    union_area = true_area + pred_area - intersect_area
    iou = intersect_area / (union_area + 1e-07)  # Добавляем epsilon для избежания деления на 0
    
    return tf.reduce_mean(iou)  # Среднее IoU по батчу

def create_model(input_shape):
    """Создание модели"""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)

    outputs = Dense(4, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=[iou_metric]
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

if __name__ == "__main__":
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

    visualize_results(model, test_images, TARGET_SIZE, 20)