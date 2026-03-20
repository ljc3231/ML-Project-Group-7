import sklearn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn(df):
    # Remove features that are entirely missing and impute remaining missing values
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)  # or use a more sophisticated imputer if desired

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    accuracy = knn_model.score(X_test, y_test)

    return accuracy

def main():
    file_path = "./Data/preprocessed/preprocessed_kddcup_10_percent.csv"
    print("cwd:", os.getcwd())
    print("file exists:", os.path.exists(file_path))
    df = pd.read_csv(file_path)
    print(knn(df))

if __name__ == "__main__":    
    main()