import os
import subprocess
import sys


def main():
    if len(sys.argv) != 2 or (sys.argv[1] != "partial" and sys.argv[1] != "full"):
        print("Unknown arguments; stopping execution")
        exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.argv[1] == "partial":
        suffix = "kddcup_10_percent.csv"
    else:
        suffix = "kddcup_full.csv"

    data_file = os.path.join(base_dir, "..", "data", suffix)

    if not os.path.exists(data_file):
        print("Running get_data.py")
        subprocess.run(["python", base_dir + "/get_data.py", sys.argv[1]])

    preprocessed_data_file = os.path.join(
        base_dir,
        "..",
        "data",
        "preprocessed",
        suffix,
    )

    if not os.path.exists(preprocessed_data_file):
        print("Running preprocessing.py")
        subprocess.run(["python", base_dir + "/preprocessing.py", sys.argv[1]])

    print("Running k_nearest.py")
    subprocess.run(["python", base_dir + "/k_nearest.py", sys.argv[1]])

    print("Running isolation_forest.py")
    subprocess.run(["python", base_dir + "/isolation_forest.py", sys.argv[1]])

    print("Running SVM")
    subprocess.run(["python", base_dir + "/svm.py", "test", f"--{sys.argv[1]}"])


if __name__ == "__main__":
    main()
