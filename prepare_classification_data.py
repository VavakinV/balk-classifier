import os
import json
import csv
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def crop_rotated_rect(image, rect):
    """Вырезает повернутый прямоугольник из изображения."""
    center, size, angle = rect
    size = (int(size[0]), int(size[1]))
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    cropped = cv2.getRectSubPix(rotated, size, center)
    return cropped

def process_annotations(annotations_path, images_path, output_dir, output_csv):
    """Обрабатывает разметку и сохраняет вырезанные изображения."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(annotations_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'producer'])
        
        for row in rows:
            image_rel_path = row['image'].split('/')[-1]
            image_path = os.path.join(images_path, image_rel_path)
            
            if not os.path.exists(image_path):
                continue
                
            bbox_data = json.loads(row['code_bbox'].replace("'", '"'))
            if not bbox_data:
                continue
                
            bbox = bbox_data[0]
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            x = bbox['x'] / 100 * bbox['original_width']
            y = bbox['y'] / 100 * bbox['original_height']
            width = bbox['width'] / 100 * bbox['original_width']
            height = bbox['height'] / 100 * bbox['original_height']
            rotation = bbox['rotation']
            
            center = (x + width/2, y + height/2)
            size = (width, height)
            
            cropped = crop_rotated_rect(img, (center, size, rotation))
            if cropped.size == 0:
                continue

            output_path = os.path.join(output_dir, f"{row['id']}.jpg")
            cv2.imwrite(output_path, cropped)

            writer.writerow([output_path, row['producer']])

if __name__ == "__main__":
    TRAIN_ANNOTATIONS = os.getenv("TRAIN_ANNOTATIONS_PATH")
    TEST_ANNOTATIONS = os.getenv("TEST_ANNOTATIONS_PATH")
    TRAIN_IMAGES = os.getenv("TRAIN_IMAGES_PATH")
    TEST_IMAGES = os.getenv("TEST_IMAGES_PATH")

    process_annotations(
        TRAIN_ANNOTATIONS,
        TRAIN_IMAGES,
        "train_cropped",
        "train_classification.csv"
    )
    
    process_annotations(
        TEST_ANNOTATIONS,
        TEST_IMAGES,
        "test_cropped",
        "test_classification.csv"
    )