import os
import argparse
from time import perf_counter

from sklearn.metrics import classification_report, accuracy_score
from common import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

GPU = False

try:
    from cuml.svm import SVC
    from cuml.model_selection import GridSearchCV
    import cudf as pd
    GPU = True
except ImportError:
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import pandas as pd

TRAIN_FULL = "train_pca_kddcup_full.csv"
TRAIN_10 = "train_pca_kddcup_10_percent.csv"
TEST_FULL = "test_pca_kddcup_full.csv"
TEST_10 = "test_pca_kddcup_10_percent.csv"
MODEL_FILE = "data/models/svm_model.pkl"

def test_and_eval(args, model, file_path):
    """
    Test a model on a given data-file and evaluate the results
    :param args: Command-line arguments
    :param model: Model to test
    :param file_path: Test data-file
    :return: None
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    y_test, y_pred = test_svm(model, df)

    y_pred = pd.Series(y_pred, name="is_anomaly").to_numpy()

    figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots", f"svm_confusion_matrix_{args.mode}.png")
    cm = generate_confusion_matrix(y_test, y_pred, figure_path, args.mode)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

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
        return None

    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1].astype(np.int32)

    X_train_cpu = X.to_pandas().values.astype('float32')
    y_train_cpu = y.to_pandas().values.astype('int32')

    base_model = SVC(kernel='rbf', class_weight='balanced')

    param_grid = {
        'C': np.logspace(-3, 3, 10),
        'gamma': np.logspace(-4, 1, 10)
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1 if GPU else -1,
        verbose=2
    )
    search.fit(X_train_cpu, y_train_cpu)
    best_model = search.best_estimator_

    scores = search.cv_results_['mean_test_score'].reshape(len(param_grid['C']),
                                                           len(param_grid['gamma']))

    plt.figure(figsize=(8, 6))
    sns.heatmap(scores, annot=True, fmt='.4f', cmap='viridis',
                vmin=scores.min(), vmax=scores.max(),
                xticklabels=[f"{g:.1e}" for g in param_grid['gamma']],
                yticklabels=[f"{c:.1e}" for c in param_grid['C']])
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.title('Grid Search F1-Score')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots", "svm_grid_search_f1_score.png"))

    print(f"Best parameters: {search.best_params_}")

    return best_model.as_sklearn() if GPU else best_model

def test_svm(svm_model, df):
    X = df.iloc[:, :-1].astype(np.float32)
    y = df.iloc[:, -1].astype(np.int64)

    y_pred = svm_model.predict(X.to_numpy())

    return y.to_numpy(), (y_pred + 1) // 2

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
        save_model(model, model_path)

        print("Finished training, testing against training data")
        test_and_eval(args, model, file_path)

    else:

        model = load_model(model_path)
        if model is None:
            return

        print(f"Testing {args.model_file} on {file_path}")
        test_and_eval(args, model, file_path)


if __name__ == "__main__":
    main()