import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_model(training_data):
    X_train = training_data.iloc[:, :-1]

    model = IsolationForest(n_estimators=100, contamination=0.05)
    model.fit(X_train)

    return model


def evaluate_model(model, testing_data):
    X_test = testing_data.iloc[:, :-1]
    y_test = testing_data.iloc[:, -1]

    y_test_transformed = np.where(y_test == 0, -1, 1)

    y_pred = model.predict(X_test)

    print("Evaluation Results (using testing data):")
    print(f"Accuracy:{accuracy_score(y_test_transformed, y_pred)}\n")
    print(
        f"Classification Report:\n{classification_report(y_test_transformed, y_pred)}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_transformed, y_pred)}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    training_data_file = os.path.join(
        base_dir, "..", "Data", "preprocessed", "train_pca_kddcup_10_percent.csv"
    )
    testing_data_file = os.path.join(
        base_dir, "..", "Data", "preprocessed", "test_pca_kddcup_10_percent.csv"
    )

    training_data = pd.read_csv(training_data_file)
    testing_data = pd.read_csv(testing_data_file)

    model = train_model(training_data)

    evaluate_model(model, testing_data)


if __name__ == "__main__":
    main()
