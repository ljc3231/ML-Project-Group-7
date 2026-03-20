from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from common import csv_to_df

def svm(df):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def main():
    file_path = "../Data/preprocessed/preprocessed_kddcup_10_percent.csv"
    df = csv_to_df(file_path)
    print(svm(df))

if __name__ == "__main__":
    main()

