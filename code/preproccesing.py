import pandas as pd
from common import csv_to_df

def main():
    file_path = '../Data/kddcup_10_percent.csv'  # Replace with your dataset path
    processed_data = csv_to_df(file_path)
    processed_data.to_csv('../Data/preprocessed/preprocessed_kddcup_10_percent.csv', index=False)  # Save processed data
    print(processed_data.head())



if __name__ == "__main__":
    main()