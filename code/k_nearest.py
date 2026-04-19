import os
import sys

import pandas as pd
from common import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

USE_PARTIAL_DATA = True

def save(model, filename):
    """
    Serialize a model to a file
    :param model: Model to save
    :param filename: Destination file
    :return: None
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def knn(train_df, test_df, k=5):
    # used before PCA, but PCA should have removed all missing values and invariant features, so this is no longer necessary
    """
    # Remove features that are entirely missing and impute remaining missing values
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    """

    y_train = train_df.iloc[:, -1]
    X_train = train_df.iloc[:, :-1]

    y_test = test_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]

    knn_model = KNeighborsClassifier(n_neighbors=k, weights="distance")

    knn_model.fit(X_train, y_train)
    save_model(knn_model, os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "models", "knn_model.pkl"))

    y_pred = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy: ", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if USE_PARTIAL_DATA:
        suffix = "kddcup_10_percent.csv"
    else:
        suffix = "kddcup_full.csv"

    train_file_path = os.path.join(
        base_dir, "..", "data", "preprocessed", "train_pca_" + suffix
    )
    test_file_path = os.path.join(
        base_dir, "..", "data", "preprocessed", "test_pca_" + suffix
    )

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # hyperparam testing
    # for k in [1, 3, 5, 7, 9]:
    # print(f"\n\nRunning KNN with k={k}")
    # print("Accuracy: ", knn(train_df, test_df, k=k))

    # N = 3 is (very) slightly worse than n = 1, but we use 3 as to prevent potential overfitting and to combat potential noise
    print("\n\nRunning KNN with k=3")
    print("Accuracy: ", knn(train_df, test_df, k=3))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "full":
            USE_PARTIAL_DATA = False
        elif sys.argv[1] != "partial":
            print("Unknown arguments; stopping execution")
            exit(1)

    main()

