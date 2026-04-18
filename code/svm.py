import os
import pickle
import argparse
from time import perf_counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from cuml.svm import SVC
from cuml.model_selection import GridSearchCV
import cudf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAIN_FULL = "train_pca_kddcup_full.csv"
TRAIN_10 = "train_pca_kddcup_10_percent.csv"
TEST_FULL = "test_pca_kddcup_full.csv"
TEST_10 = "test_pca_kddcup_10_percent.csv"
MODEL_FILE = "data/models/svm_model.pkl"

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

def test_and_eval(model, file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    y_test, y_pred = test_svm(model, df)

    y_pred = pd.Series(y_pred, name="is_anomaly")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))



def train_svm(file_path):
    """
    Train a support vector machine based on a given data-file
    :param file_path: Data-file
    :return: Trained model
    """
    print(f"Training SVM on {file_path}")
    try:
        df = cudf.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1].astype(np.int32)

    base_model = SVC(kernel='rbf', class_weight='balanced')

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001]
    }

    X_train_cpu = X.to_pandas().values.astype('float32')
    y_train_cpu = y.to_pandas().values.astype('int32')

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=2
    )
    search.fit(X_train_cpu, y_train_cpu)
    best_model = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")

    return best_model.as_sklearn()

def test_svm(svm_model, df):
    """
    Test given model against given test-data
    :param svm_model: Pre-trained model
    :param df: Dataframe containing test-data
    :return: test labels and predicted labels
    """
    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1].astype(np.int32)

    y_pred = svm_model.predict(X.values)
    return y, (y_pred + 1) // 2

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
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        print(f"{filename} is empty or corrupted.")
        return None
    except FileNotFoundError:
        print(f"Model file not found: {filename}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train or test an SVM model")
    parser.add_argument("mode", choices=["train", "test"], help="Run mode")
    parser.add_argument("model_file", nargs="?", default=MODEL_FILE, help="Model file path")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--partial", action="store_true", help="Use partial dataset")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "train":
        file_name = TRAIN_FULL if args.full else TRAIN_10
    else:
        file_name = TEST_FULL if args.full else TEST_10

    file_path = os.path.join(base_dir, '..', 'data', 'preprocessed', file_name)
    model_path = os.path.join(base_dir, '..', args.model_file)

    if args.mode == "train":
        start = perf_counter()
        model = train_svm(file_path)
        print(output_timing(perf_counter() - start))
        save_svm(model, model_path)

        print("Finished training, testing against training data")
        test_and_eval(model, file_path)

    else:

        model = load_svm(model_path)
        if model is None:
            return

        print(f"Testing {args.model_file} on {file_path}")
        test_and_eval(model, file_path)


if __name__ == "__main__":
    main()