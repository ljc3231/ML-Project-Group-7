import os
import sys

import numpy as np
import pandas as pd
from common import *
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report

USE_PARTIAL_DATA = True


def train_model(training_data):
    X_train = training_data.iloc[:, :-1]

    model = IsolationForest(n_estimators=100, contamination=0.2)
    model.fit(X_train)
    save_model(
        model,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "models",
            "iso_model.pkl",
        ),
    )

    return model


def evaluate_model(model, testing_data):
    X_test = testing_data.iloc[:, :-1]
    y_test = testing_data.iloc[:, -1]

    y_test_transformed = np.where(y_test == 0, -1, 1)

    y_pred = model.predict(X_test)

    print("Evaluation Results (using testing data):")
    print(f"Accuracy: {accuracy_score(y_test_transformed, y_pred)}\n")
    print(
        f"Classification Report:\n{classification_report(y_test_transformed, y_pred)}"
    )
    figure_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "plots",
        "iso_confusion_matrix.png",
    )
    cm = generate_confusion_matrix(y_test_transformed, y_pred, figure_path, "test")
    print(f"Confusion Matrix:\n{cm}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if USE_PARTIAL_DATA:
        suffix = "kddcup_10_percent.csv"
    else:
        suffix = "kddcup_full.csv"

    training_data_file = os.path.join(
        base_dir, "..", "data", "preprocessed", "train_pca_" + suffix
    )
    testing_data_file = os.path.join(
        base_dir, "..", "data", "preprocessed", "test_pca_" + suffix
    )

    training_data = pd.read_csv(training_data_file)
    testing_data = pd.read_csv(testing_data_file)

    model = train_model(training_data)

    evaluate_model(model, testing_data)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "full":
            USE_PARTIAL_DATA = False
        elif sys.argv[1] != "partial":
            print("Unknown arguments; stopping execution")
            exit(1)

    main()
