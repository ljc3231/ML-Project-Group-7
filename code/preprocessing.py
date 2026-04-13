import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'Data', 'kddcup_10_percent.csv')

    processed_data = csv_to_df(file_path)
    processed_data = reject_invariant_features(processed_data)

    X = processed_data.drop(columns=['is_anomaly'])
    y = processed_data['is_anomaly']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify = y)

    pipeline = Pipeline(steps=[
        ('pca', PCA()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        'pca__n_components': [0.9, 0.95, 0.99],
        'pca__whiten': [False, True]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid = param_grid,
        cv = 5,
        scoring = 'f1',
        n_jobs = -1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("Test set results:")
    print(classification_report(y_test, y_pred))

    pipeline_no_clf = Pipeline(best_model.steps[:-1])
    X_train_pca = pipeline_no_clf.transform(X_train)
    X_test_pca = pipeline_no_clf.transform(X_test)

    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)

    X_train_pca_df['is_anomaly'] = y_train.reset_index(drop=True)
    X_test_pca_df['is_anomaly'] = y_test.reset_index(drop=True)

    output_dir = os.path.join(base_dir, '..', 'Data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    X_train_pca_df.to_csv(os.path.join(output_dir, 'train_pca_kddcup_10_percent.csv'), index=False)
    X_test_pca_df.to_csv(os.path.join(output_dir, 'test_pca_kddcup_10_percent.csv'), index=False)
    
    print("\nPCA-transformed training data:")
    print(X_train_pca_df.head())



if __name__ == "__main__":
    main()
