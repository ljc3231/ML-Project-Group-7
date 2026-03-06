import os
import gzip
import shutil
import pandas as pd
from sklearn.datasets import fetch_kddcup99

#DATA IS BIG, WILL TAKE TIME TO RUN!!!!!! -Liam

# Create Data directory if it doesn't exist

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "../Data")
os.makedirs(data_dir, exist_ok=True)

# Save SKLearn dataset to CSV

def save_sklearn_kdd(percent10=False, subset=None, filename="output.csv"):
    print(f"\nDownloading dataset → {filename}")
    
    dataset = fetch_kddcup99(
        percent10=percent10,
        subset=subset,
        data_home=data_dir,
        download_if_missing=True
    )

    # Convert numpy arrays to string safely
    X = dataset.data.astype(str)
    y = dataset.target.astype(str)

    df = pd.DataFrame(X)
    df["label"] = y

    path = os.path.join(data_dir, filename)
    df.to_csv(path, index=False)

    print(f"Saved to ./Data/{filename}")
    print("Shape:", df.shape)


#Download and save datasets

# 10% Training Dataset
save_sklearn_kdd(
    percent10=True,
    filename="kddcup_10_percent.csv"
)

# Full Training Dataset (~4.9 million rows)
save_sklearn_kdd(
    percent10=False,
    filename="kddcup_full.csv"
)

# Corrected Test Dataset
#save_sklearn_kdd(
#    percent10=False,
#    subset="corrected",
#    filename="kddcup_corrected_test.csv"
#)

print("\nAll datasets downloaded successfully.")
