from train_all_models import train_all_models
from fullmodel_test import run_tests

if __name__ == "__main__":
    train_models = input("Train models?\n('No' -> Enter, 'Yes' -> <any key>+Enter): ")
    if train_models:
        train_all_models()
    run_tests()
