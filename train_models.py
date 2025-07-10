import os
from torch import device, cuda
from dotenv import load_dotenv
from bbox_trainmodel import train_model as train_bbox_model
from classification_trainmodel import train_model_cropped as train_classification_model_cropped
from classification_trainmodel import train_model_full as train_classification_model_full
from loaders import load_data
from bbox_visualization import visualize_test_predictions

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 120
DEVICE = device('cuda' if cuda.is_available() else 'cpu')

TRAIN_ANNOTATIONS_PATH = os.getenv("TRAIN_ANNOTATIONS_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")
TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")

def train_models(choice=0):
    # Обучение моделей
    # Обученные модели сохраняются в файлах
    if choice == 1 or choice == 3:
        bbox_model = train_bbox_model().to(DEVICE)
    if choice == 2 or choice == 3:
        classification_model_cropped = train_classification_model_cropped()
        classification_model_full = train_classification_model_full()
    return
    # Визуализация первой модели
    # visualize_test_predictions(trained_model, test_data, TARGET_SIZE, DEVICE, n=10)