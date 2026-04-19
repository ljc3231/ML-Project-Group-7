import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MINIMUM_CROSS_CORRELATION = 0.001
USE_PARTIAL_DATA = True


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

    if USE_PARTIAL_DATA:
        suffix = "kddcup_10_percent.csv"
    else:
        suffix = "kddcup_full.csv"

    file_path = os.path.join(base_dir, "..", "data", suffix)

    processed_data = csv_to_df(file_path)
    processed_data = reject_invariant_features(processed_data)

    X = processed_data.drop(columns=["is_anomaly"])
    y = pd.DataFrame(processed_data["is_anomaly"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    train_df = pd.DataFrame(X_train_pca)
    train_df["is_anomaly"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_pca)
    test_df["is_anomaly"] = y_test.reset_index(drop=True)

    output_dir = os.path.join(base_dir, "..", "data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train_pca_" + suffix), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_pca_" + suffix), index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "full":
            USE_PARTIAL_DATA = False
        elif sys.argv[1] != "partial":
            print("Unknown arguments; stopping execution")
            exit(1)

    main()
