import os
from torch import device, cuda
from dotenv import load_dotenv
from bboxtrainmodel import train_model, calculate_test_iou
from loaders import load_data
from bboxvisualization import visualize_test_predictions

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = device('cuda' if cuda.is_available() else 'cpu')

TRAIN_ANNOTATIONS_PATH = os.getenv("TRAIN_ANNOTATIONS_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")
TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")

if __name__ == "__main__":
    # Обучение модели
    trained_model = train_model().to(DEVICE)
    
    # Загрузка тестовых данных
    test_data = load_data(TEST_ANNOTATIONS_PATH, TEST_IMAGES_PATH)
    print(f"Loaded {len(test_data)} test samples")
    
    # Расчет средней IoU для тестового набора
    test_iou = calculate_test_iou(trained_model, test_data, DEVICE)
    print(f"Mean IoU on test set: {test_iou:.4f}")
    
    # Визуализация результатов
    visualize_test_predictions(trained_model, test_data, TARGET_SIZE, DEVICE, n=10)