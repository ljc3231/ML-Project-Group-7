import os
import pickle
import sys
from time import perf_counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_FULL = "train_pca_kddcup_full.csv"
TRAIN_10 = "train_pca_kddcup_10_percent.csv"
TEST_FULL = "test_pca_kddcup_full.csv"
TEST_10 = "test_pca_kddcup_10_percent.csv"
MODEL_FILE = "models/svm_model.pkl"

def output_timing(seconds):
    """
    Transforms seconds into a human-readable string
    :param seconds: Number of seconds
    :return: Human-readable string
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    result = ""
    result += f"{int(hours)} hours" if hours > 0 else ""
    result += f", {int(minutes)} minutes" if minutes > 0 else ""
    result += f", {int(seconds)} seconds"
    return result


def train_svm(file_path):
    """
    Train a support vector machine based on a given data-file
    :param file_path: Data-file
    :return: Trained model
    """
    print(f"Training SVM on {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    if not verify_save_location(MODEL_FILE):
        print(f"Model file not found: {MODEL_FILE}")
        return

    X = df.iloc[:, :-1]

    svm_model = OneClassSVM(kernel='rbf', gamma='auto')
    svm_model.fit(X)

    return svm_model

def test_svm(svm_model, df):
    """
    Test given model against given test-data
    :param svm_model: Pre-trained model
    :param df: Dataframe containing test-data
    :return: test labels and predicted labels
    """
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    y_pred = svm_model.predict(X)
    return y, y_pred

def verify_save_location(filename):
    """
    Verify a location can be written to
    :param filename: Path to test
    :return: Whether the location can be written to
    """
    try:
        with open(filename, 'wb') as _:
            return True
    except FileNotFoundError:
        return False

def save_svm(svm_model, filename):
    """
    Serialize a model to a file
    :param svm_model: Model to save
    :param filename: Destination file
    :return: None
    """
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)

def load_svm(filename):
    """
    Load a model from a file
    :param filename: File to load
    :return: Loaded model
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4 or  sys.argv[1] not in ["train", "test"]:
        print("Usage: python svm.py <train|test> [--full] [model_file]")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    file_index = 2
    if len(sys.argv) == 4 and sys.argv[2] == "--full":
        file_path = os.path.join(base_dir, '..', 'Data', 'preprocessed', TRAIN_FULL if sys.argv[1] == "train" else TEST_FULL)
        file_index += 1
    else:
        file_path = os.path.join(base_dir, '..', 'Data', 'preprocessed', TRAIN_10 if sys.argv[1] == "train" else TEST_10)

    if sys.argv[1] == "train":
        start = perf_counter()
        model = train_svm(file_path)
        print(output_timing(perf_counter() - start))
        save_svm(model, MODEL_FILE)
    else:
        print(f"Testing {sys.argv[file_index]}")
        df = pd.read_csv(file_path)
        model = load_svm(os.path.join(base_dir, sys.argv[file_index]))
        y_test, y_pred = test_svm(model, df)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        print("Classification Report:")
        print(classification_report(y_test, y_pred))




if __name__ == "__main__":
    main()

