import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

def csv_to_df(file_path):
    df = pd.read_csv(file_path)

    target_name = df.columns[-1]
    y = df[target_name].eq("normal").astype(int)

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
    # X_processed = X_processed.fillna(0);

    y_series = pd.Series(y, index=df.index, name="is_anomaly")

    return pd.concat([X_processed, y_series], axis=1)

def calc_cross_correlation(df):
    cc_df = df.copy()
    cc_df = cc_df.drop(columns=['19', '20'])
    cc_df = cc_df.corr('pearson')

    return cc_df

def main():
    file_path = '../Data/kddcup_10_percent.csv'  # Replace with your dataset path
    processed_data = csv_to_df(file_path)
    processed_data.to_csv('../Data/preprocessed/preprocessed_kddcup_10_percent.csv', index=False)  # Save processed data
    print(processed_data.head())
    calc_cross_correlation(processed_data)



if __name__ == "__main__":
    main()
