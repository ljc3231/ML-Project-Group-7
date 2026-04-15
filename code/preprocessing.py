import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MINIMUM_CROSS_CORRELATION = 0.001


def csv_to_df(file_path):
    df = pd.read_csv(file_path)

    target_name = df.columns[-1]
    y = (df[target_name] != "normal.").astype(int)

    X = df.drop(columns=[target_name])

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=False, drop="first", handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include=["number"]),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=["object"]),
            ),
        ],
        verbose_feature_names_out=False,
    )

    X_transformed = preprocessor.fit_transform(X)

    column_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_transformed, columns=column_names, index=df.index)

    y_series = pd.Series(y, index=df.index, name="is_anomaly")

    return pd.concat([X_processed, y_series], axis=1)


def reject_invariant_features(df):
    cc_df = df.corr("pearson")
    cc_df = cc_df.fillna(0)

    mask = (cc_df["is_anomaly"] < MINIMUM_CROSS_CORRELATION) & (
        cc_df["is_anomaly"] > -MINIMUM_CROSS_CORRELATION
    )
    mask = mask.to_list()
    indices = cc_df.index.to_list()
    invariant_features = [indices for indices, mask in zip(indices, mask) if mask]

    df.drop(columns=invariant_features, inplace=True)

    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "Data", "kddcup_10_percent.csv")

    processed_data = csv_to_df(file_path)
    processed_data = reject_invariant_features(processed_data)

    X = processed_data.drop(columns=["is_anomaly"])
    y = pd.DataFrame(processed_data["is_anomaly"])

    pca = PCA(n_components=12)
    X_pca = pd.DataFrame(pca.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y
    )

    training_data = pd.DataFrame(X_train)
    training_data["is_anomaly"] = y_train

    testing_data = pd.DataFrame(X_test)
    testing_data["is_anomaly"] = y_test

    output_dir = os.path.join(base_dir, "..", "Data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    training_data.to_csv(
        os.path.join(output_dir, "train_pca_kddcup_10_percent.csv"), index=False
    )
    testing_data.to_csv(
        os.path.join(output_dir, "test_pca_kddcup_10_percent.csv"), index=False
    )


if __name__ == "__main__":
    main()
