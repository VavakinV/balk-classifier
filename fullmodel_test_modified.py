import os
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from predict import FullPipeline
from collections import defaultdict, Counter

load_dotenv()

DETECTOR_MODEL_PATH = "best_model.pth"
CLASSIFIER_CROPPED_PATH = "producer_classifier.pth"
CLASSIFIER_FULL_PATH = "producer_classifier_full.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODIFIED_TEST_IMAGES_PATH = os.getenv("MODIFIED_TEST_IMAGES_PATH")


def load_test_data(images_root):
    """
    Проходит по структуре:
      images_root/
        producer_1/
          half/
          full/
        producer_2/
          half/
          full/
      ...
    Возвращает список кортежей (image_path, producer, variant)
    где variant — 'half' или 'full'.
    """
    data = []
    # Перебор производителей
    for producer in sorted(os.listdir(images_root)):
        producer_dir = os.path.join(images_root, producer)
        if not os.path.isdir(producer_dir):
            continue
        # Перебор вариантов изображения
        for variant in ['half', 'full']:
            variant_dir = os.path.join(producer_dir, variant)
            if not os.path.isdir(variant_dir):
                continue
            # Перебор файлов изображений
            for fname in sorted(os.listdir(variant_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(variant_dir, fname)
                    data.append((img_path, producer, variant))
    return data


def evaluate_pipeline(test_items):
    """
    test_items: список (image_path, true_producer, variant)
    Возвращает словари с точностью по производителю и по варианту (half/full).
    """
    pipeline = FullPipeline(
        detector_path=DETECTOR_MODEL_PATH,
        classifier_cropped_path=CLASSIFIER_CROPPED_PATH,
        classifier_full_path=CLASSIFIER_FULL_PATH,
        threshold=1.1,
        device=DEVICE
    )

    # Счётчики
    producer_total = Counter()
    producer_correct = Counter()
    variant_total = Counter()
    variant_correct = Counter()

    for img_path, true_prod, variant in tqdm(test_items, desc="Processing images"):
        try:
            result = pipeline.predict(img_path)
            pred = result['producer']

            # Обновляем общие счётчики
            producer_total[true_prod] += 1
            variant_total[variant] += 1

            if pred == true_prod:
                producer_correct[true_prod] += 1
                variant_correct[variant] += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Вычисление точностей
    per_producer_accuracy = {
        prod: (producer_correct[prod] / producer_total[prod] * 100 if producer_total[prod] > 0 else 0)
        for prod in sorted(producer_total)
    }
    per_variant_accuracy = {
        var: (variant_correct[var] / variant_total[var] * 100 if variant_total[var] > 0 else 0)
        for var in ['half', 'full']
    }

    overall_total = sum(producer_total.values())
    overall_correct = sum(producer_correct.values())
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0

    return {
        'overall_accuracy': overall_accuracy,
        'per_producer_accuracy': per_producer_accuracy,
        'per_variant_accuracy': per_variant_accuracy,
        'producer_total': producer_total,
        'producer_correct': producer_correct,
        'variant_total': variant_total,
        'variant_correct': variant_correct,
        'overall_total': overall_total,
        'overall_correct': overall_correct
    }


def print_results(stats):
    print("\n=== Overall Accuracy ===")
    print(f"Total images: {stats['overall_total']}")
    print(f"Correct: {stats['overall_correct']}")
    print(f"Accuracy: {stats['overall_accuracy']:.2f}%")

    print("\n=== Accuracy per Producer ===")
    for prod, acc in stats['per_producer_accuracy'].items():
        total = stats['producer_total'][prod]
        correct = stats['producer_correct'][prod]
        print(f"{prod:<10} | {acc:6.2f}% ({correct}/{total})")

    print("\n=== Accuracy per Variant ===")
    for var, acc in stats['per_variant_accuracy'].items():
        total = stats['variant_total'][var]
        correct = stats['variant_correct'][var]
        print(f"{var:<5} | {acc:6.2f}% ({correct}/{total})")


def run_tests():
    if not MODIFIED_TEST_IMAGES_PATH:
        raise RuntimeError("MODIFIED_TEST_IMAGES_PATH is not set in environment variables")
    test_items = load_test_data(MODIFIED_TEST_IMAGES_PATH)
    print(f"Loaded {len(test_items)} test images from {MODIFIED_TEST_IMAGES_PATH}")

    stats = evaluate_pipeline(test_items)
    print_results(stats)


if __name__ == '__main__':
    run_tests()
