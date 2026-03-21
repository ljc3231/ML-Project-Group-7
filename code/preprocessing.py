import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

MINIMUM_CROSS_CORRELATION = 0.001

def csv_to_df(file_path):
    df = pd.read_csv(file_path)

    target_name = df.columns[-1]
    y = (df[target_name] != "normal.").astype(int)

    X = df.drop(columns=[target_name])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=['number'])),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object']))
        ],
        verbose_feature_names_out=False
    )

    X_transformed = preprocessor.fit_transform(X)

    column_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_transformed, columns=column_names, index=df.index)

    y_series = pd.Series(y, index=df.index, name="is_anomaly")

    return pd.concat([X_processed, y_series], axis=1)

def reject_invariant_features(df):
    cc_df = df.corr('pearson')
    cc_df = cc_df.fillna(0)

    mask = (cc_df['is_anomaly'] < MINIMUM_CROSS_CORRELATION) & (cc_df['is_anomaly'] > -MINIMUM_CROSS_CORRELATION)
    mask = mask.to_list()
    indices = cc_df.index.to_list()
    invariant_features = [indices for indices, mask in zip(indices, mask) if mask]

    df.drop(columns=invariant_features, inplace=True)

    return df

def main():
    file_path = '../Data/kddcup_10_percent.csv'  # Replace with your dataset path
    processed_data = csv_to_df(file_path)
    processed_data = reject_invariant_features(processed_data)
    processed_data.to_csv('../Data/preprocessed/preprocessed_kddcup_10_percent.csv', index=False)  # Save processed data
    print(processed_data.head())



if __name__ == "__main__":
    main()
