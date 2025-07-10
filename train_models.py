from torch import device, cuda
from bbox_trainmodel import train_model as train_bbox_model
from classification_trainmodel import train_model_cropped as train_classification_model_cropped
from classification_trainmodel import train_model_full as train_classification_model_full

DEVICE = device('cuda' if cuda.is_available() else 'cpu')


def train_models(choice=0):
    # Обучение моделей
    # Обученные модели сохраняются в файлах
    if choice == 1 or choice == 3:
        bbox_model = train_bbox_model().to(DEVICE)
    if choice == 2 or choice == 3:
        classification_model_cropped = train_classification_model_cropped()
        classification_model_full = train_classification_model_full()
    return