import pickle
import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import pandas as pd

"""
Currently assumes last row is the label, our data doesn't really lend itself to anomaly detection if we are trying to
predict which particular anomaly occurred so One-Class SVM may not work with this format.
"""
def train_svm(df):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.01)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def save_svm(svm_model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)

def load_svm(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    file_path = "../Data/preprocessed/preprocessed_kddcup_10_percent.csv"
    if sys.argv[0] == "train":
        df = pd.read_csv(file_path)
        model = train_svm(df)
        save_svm(model, sys.argv[1])
    else:
        model = load_svm(sys.argv[1])

    # do stuff with the model


if __name__ == "__main__":
    main()

