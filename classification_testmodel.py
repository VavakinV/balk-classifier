import torch
from torch.utils.data import DataLoader
from classification_model import ProducerClassifier
from classification_dataset import ProducerDataset
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

model = ProducerClassifier(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load("producer_classifier.pth"))
model.eval()

test_dataset = ProducerDataset("test_classification.csv")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")