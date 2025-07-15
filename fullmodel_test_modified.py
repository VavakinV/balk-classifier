import os
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from predict import FullPipeline
from collections import Counter

load_dotenv()

DETECTOR_MODEL_PATH = "best_model.pth"
CLASSIFIER_CROPPED_PATH = "producer_classifier.pth"
CLASSIFIER_FULL_PATH = "producer_classifier_full.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODIFIED_TEST_IMAGES_PATH = os.getenv("MODIFIED_TEST_IMAGES_PATH")


def load_test_data(images_root):
    """
    Собирает тестовые элементы из структуры:
      images_root/
        producer_1/
          half/
          full/
        ...
    Возвращает список кортежей (image_path, producer, variant)
    где variant — 'half' или 'full'.
    """
    data = []
    for producer in sorted(os.listdir(images_root)):
        producer_dir = os.path.join(images_root, producer)
        if not os.path.isdir(producer_dir):
            continue
        for variant in ['half', 'full']:
            variant_dir = os.path.join(producer_dir, variant)
            if not os.path.isdir(variant_dir):
                continue
            for fname in sorted(os.listdir(variant_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(variant_dir, fname)
                    data.append((img_path, producer, variant))
    return data


def evaluate_pipeline(test_items):
    """
    test_items: список (image_path, true_producer, variant)
    Возвращает счётчики для подсчёта точностей.
    """
    pipeline = FullPipeline(
        detector_path=DETECTOR_MODEL_PATH,
        classifier_cropped_path=CLASSIFIER_CROPPED_PATH,
        classifier_full_path=CLASSIFIER_FULL_PATH,
        threshold=1.1,
        device=DEVICE
    )

    producer_total = Counter()
    producer_correct = Counter()
    prod_var_total = Counter()
    prod_var_correct = Counter()

    for img_path, true_prod, variant in tqdm(test_items, desc="Processing images"):
        try:
            result = pipeline.predict(img_path)
            pred = result['producer']

            producer_total[true_prod] += 1
            prod_var_total[(true_prod, variant)] += 1

            if pred == true_prod:
                producer_correct[true_prod] += 1
                prod_var_correct[(true_prod, variant)] += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return producer_total, producer_correct, prod_var_total, prod_var_correct


def print_results(producer_total, producer_correct, prod_var_total, prod_var_correct):
    """
    Выводит таблицу с колонками: производитель | точность half | точность full | общая точность
    """
    producers = sorted(producer_total.keys())
    print(f"{'Producer':<15} {'Half Acc (%)':<15} {'Full Acc (%)':<15} {'Overall Acc (%)':<15}")
    for prod in producers:
        total_half = prod_var_total.get((prod, 'half'), 0)
        correct_half = prod_var_correct.get((prod, 'half'), 0)
        acc_half = (correct_half / total_half * 100) if total_half else 0

        total_full = prod_var_total.get((prod, 'full'), 0)
        correct_full = prod_var_correct.get((prod, 'full'), 0)
        acc_full = (correct_full / total_full * 100) if total_full else 0

        total_all = producer_total[prod]
        correct_all = producer_correct[prod]
        acc_all = (correct_all / total_all * 100) if total_all else 0

        print(f"{prod:<15} {acc_half:<15.2f} {acc_full:<15.2f} {acc_all:<15.2f}")


def run_tests():
    test_items = load_test_data(MODIFIED_TEST_IMAGES_PATH)
    print(f"Loaded {len(test_items)} test images")

    producer_total, producer_correct, prod_var_total, prod_var_correct = evaluate_pipeline(test_items)

    print_results(producer_total, producer_correct, prod_var_total, prod_var_correct)


if __name__ == '__main__':
    run_tests()
