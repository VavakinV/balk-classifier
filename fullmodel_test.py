import os
import csv
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict, Counter

from predict import FullPipeline

load_dotenv()

TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")

DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH")  # YOLO weights
CLASSIFIER_CROPPED_PATH = "producer_classifier.pth"
CLASSIFIER_FULL_PATH = "producer_classifier_full.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_data(annotations_path, images_path):
    """
    Загружает тестовые данные:
    { image_path : true_producer }
    """
    data = {}

    with open(annotations_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = os.path.basename(row["image"])
            image_path = os.path.join(images_path, image_name)

            if os.path.exists(image_path):
                data[image_path] = row["producer"]

    return data


def evaluate_pipeline(test_data):
    pipeline = FullPipeline(
        detector_weights_path=DETECTOR_MODEL_PATH,
        classifier_cropped_path=CLASSIFIER_CROPPED_PATH,
        classifier_full_path=CLASSIFIER_FULL_PATH,
        threshold=0.8, # САМЫЙ ВАЖНЫЙ ПАРАМЕТР В МИРЕ
        device=DEVICE
    )

    results = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    producer_correct = Counter()
    producer_total = Counter()

    for image_path, true_producer in tqdm(
        test_data.items(),
        desc="Processing images"
    ):
        try:
            result = pipeline.predict(image_path)

            pred_producer = result["producer"]
            confidence = result["confidence"]
            is_correct = pred_producer == true_producer

            results.append({
                "image": os.path.basename(image_path),
                "true_producer": true_producer,
                "pred_producer": pred_producer,
                "confidence": confidence,
                "correct": is_correct
            })

            producer_total[true_producer] += 1
            if is_correct:
                producer_correct[true_producer] += 1

            confusion_matrix[true_producer][pred_producer] += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    total = sum(producer_total.values())
    correct = sum(producer_correct.values())
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "results": results,
        "confusion_matrix": confusion_matrix,
        "producer_total": producer_total,
        "producer_correct": producer_correct,
        "total": total,
        "correct": correct
    }


def print_results(eval_results):
    print("\n=== Detailed Predictions ===")
    for res in eval_results["results"]:
        status = "✓" if res["correct"] else "✗"
        print(
            f"{status} Image: {res['image']:<30} | "
            f"True: {res['true_producer']:<12} | "
            f"Pred: {res['pred_producer']:<12} | "
            f"Conf: {res['confidence']:.2f}"
        )

    print("\n=== Summary ===")
    print(f"Total images: {eval_results['total']}")
    print(f"Correct predictions: {eval_results['correct']}")
    print(f"Overall Accuracy: {eval_results['accuracy']:.2f}%")


    print("\n=== Per-Producer Accuracy ===")
    for producer in sorted(eval_results["producer_total"].keys()):
        total = eval_results["producer_total"][producer]
        correct = eval_results["producer_correct"][producer]
        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"{producer:<12} | Accuracy: {acc:.2f}% ({correct}/{total})")

    print("\n=== Confusion Matrix ===")
    true_labels = sorted(eval_results["confusion_matrix"].keys())
    pred_labels = sorted({
        p for t in true_labels for p in eval_results["confusion_matrix"][t].keys()
    })

    print(f"{'True\\Pred':<15}" + "".join(f"{p:<15}" for p in pred_labels))
    for true in true_labels:
        row = "".join(
            f"{eval_results['confusion_matrix'][true].get(pred, 0):<15}"
            for pred in pred_labels
        )
        print(f"{true:<15}{row}")


def run_tests():
    test_data = load_test_data(TEST_ANNOTATIONS_PATH, TEST_IMAGES_PATH)
    print(f"Loaded {len(test_data)} test images")

    eval_results = evaluate_pipeline(test_data)
    print_results(eval_results)
