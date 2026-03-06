import pandas as pd

def csv_to_df(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    #missing value -> mean of col
    df.fillna(df.mean(numeric_only=True), inplace=True)

    #one-hot
    df = pd.get_dummies(df, drop_first=True)

    #normalizaton
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    print(df.info())  # Print dataset info
    return df

def main():
    file_path = '../Data/kddcup_10_percent.csv'  # Replace with your dataset path
    processed_data = csv_to_df(file_path)
    processed_data.to_csv('./Data/preprocessed/preprocessed_kddcup_10_percent.csv', index=False)  # Save processed data
    print(processed_data.head())



if __name__ == "__main__":
    main()