from train_models import train_models
from fullmodel_test import run_tests

if __name__ == "__main__":
    choice = int(input("Which model to train?\n1 -> Only bbox\n2 -> Only classification\n3 -> Both\n0 -> None\nChoose: "))
    try:
        train_models(choice)
        run_tests()
    except Exception as e:
        print(f"Error: {e}")
    
