import pickle
import sys

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import pandas as pd

"""
Train a support vector machine based on a given dataframe
"""
def train_svm(df):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = OneClassSVM(kernel='rbf', gamma='auto')
    svm_model.fit(X_train, y_train)

    return svm_model

def test_svm(svm_model, df):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = svm_model.predict(X_test)
    return y_test, y_pred


def save_svm(svm_model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)

def load_svm(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4 or  sys.argv[1] not in ["train", "test"]:
        print("Usage: python svm.py <train|test> [--full] <model_file>")
        return

    file_index = 2
    if len(sys.argv) == 4 and sys.argv[2] == "--full":
        file_path = "Data/preprocessed/preprocessed_kddcup_full.csv"
        file_index += 1
    else:
        file_path = "/home/dvinciulla/Documents/ML/Project/Data/preprocessed/preprocessed_kddcup_10_percent.csv"

    if sys.argv[1] == "train":
        print("Training SVM...")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return
        model = train_svm(df)
        save_svm(model, "../models/svm_model.pkl")
    else:
        print(f"Testing {sys.argv[file_index]}")
        df = pd.read_csv(file_path)
        model = load_svm(sys.argv[file_index])
        y_test, y_pred = test_svm(model, df)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"F1: {f1_score(y_test, y_pred)}")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()




if __name__ == "__main__":
    main()

