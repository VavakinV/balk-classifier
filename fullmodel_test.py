import os
import csv
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from predict import FullPipeline
from collections import defaultdict, Counter

load_dotenv()

TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")
DETECTOR_MODEL_PATH = "best_model.pth"
CLASSIFIER_CROPPED_PATH = "producer_classifier.pth"
CLASSIFIER_FULL_PATH = "producer_classifier_full.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_test_data(annotations_path, images_path):
    data = {}
    with open(annotations_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = os.path.basename(row['image'])
            image_path = os.path.join(images_path, image_name)
            if os.path.exists(image_path):
                data[image_path] = row['producer']
    return data

def evaluate_pipeline(test_data):
    pipeline = FullPipeline(
        detector_path=DETECTOR_MODEL_PATH,
        classifier_cropped_path=CLASSIFIER_CROPPED_PATH,
        classifier_full_path=CLASSIFIER_FULL_PATH,
        threshold=1.1,
        device=DEVICE
    )

    results = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    producer_correct = Counter()
    producer_total = Counter()

    for image_path, true_producer in tqdm(test_data.items(), desc="Processing images"):
        try:
            result = pipeline.predict(image_path)
            pred_producer = result['producer']
            confidence = result['confidence']

            is_correct = (pred_producer == true_producer)
            results.append({
                'image': os.path.basename(image_path),
                'true_producer': true_producer,
                'pred_producer': pred_producer,
                'confidence': confidence,
                'correct': is_correct
            })

            producer_total[true_producer] += 1
            if is_correct:
                producer_correct[true_producer] += 1

            confusion_matrix[true_producer][pred_producer] += 1

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    total = sum(producer_total.values())
    correct = sum(producer_correct.values())
    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        'accuracy': accuracy,
        'results': results,
        'confusion_matrix': confusion_matrix,
        'producer_total': producer_total,
        'producer_correct': producer_correct,
        'total': total,
        'correct': correct
    }

def print_results(eval_results):
    print("\n=== Detailed Predictions ===")
    for res in eval_results['results']:
        status = "✓" if res['correct'] else "✗"
        print(f"{status} Image: {res['image']:<30} | True: {res['true_producer']:<8} | "
              f"Pred: {res['pred_producer']:<8} | Conf: {res['confidence']:.2f}")

    print("\n=== Summary ===")
    print(f"Total images: {eval_results['total']}")
    print(f"Correct predictions: {eval_results['correct']}")
    print(f"Overall Accuracy: {eval_results['accuracy']:.2f}%")

    print("\n=== Per-Producer Accuracy ===")
    for producer in sorted(eval_results['producer_total'].keys()):
        total = eval_results['producer_total'][producer]
        correct = eval_results['producer_correct'][producer]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"{producer:<10} | Accuracy: {accuracy:.2f}% ({correct}/{total})")

    print("\n=== Confusion Matrix ===")
    true_producers = sorted(eval_results['confusion_matrix'].keys())
    pred_producers = sorted(set(p for true in true_producers for p in eval_results['confusion_matrix'][true].keys()))

    print(f"{'True\\Pred':<12}" + ''.join(f"{p:<12}" for p in pred_producers))
    for true in true_producers:
        print(f"{true:<12}" + ''.join(f"{eval_results['confusion_matrix'][true].get(pred, 0):<12}" for pred in pred_producers))

def run_tests():
    test_data = load_test_data(TEST_ANNOTATIONS_PATH, TEST_IMAGES_PATH)
    print(f"Loaded {len(test_data)} test images")

    eval_results = evaluate_pipeline(test_data)
    print_results(eval_results)