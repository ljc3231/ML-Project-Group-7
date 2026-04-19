import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from common import load_model


def roc_auc(suffix):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    models = {
        "K-Nearest Neighbors": load_model(os.path.join(base_dir, "..", "data", "models", "knn_model.pkl")),
        "SVM": load_model(os.path.join(base_dir, "..", "data", "models", "svm_model.pkl"))
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    df = pd.read_csv(os.path.join(base_dir, "..", "data", "preprocessed", "test_pca_" + suffix))
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    iso = load_model(os.path.join(base_dir, "..", "data", "models", "iso_model.pkl"))
    iso_scores = iso.decision_function(X_test)

    for name, clf in models.items():
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name=name)

    RocCurveDisplay.from_predictions(np.where(y_test == 0, -1, 1), -iso_scores.ravel(), ax=ax, name="Isolation Forest")

    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Chance (AUC = 0.5)")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def prdisplay(suffix):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(base_dir, "..", "data", "preprocessed", "test_pca_" + suffix))
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]

    knn = load_model(os.path.join(base_dir, "..", "data", "models", "knn_model.pkl"))
    svm = load_model(os.path.join(base_dir, "..", "data", "models", "svm_model.pkl"))
    iso = load_model(os.path.join(base_dir, "..", "data", "models", "iso_model.pkl"))


    fig, ax = plt.subplots(figsize=(8, 6))

    # Use from_estimator for standard classifiers
    PrecisionRecallDisplay.from_estimator(knn, X_test, y_test, ax=ax, name="KNN")
    PrecisionRecallDisplay.from_estimator(svm, X_test, y_test, ax=ax, name="SVM")

    # Use from_predictions for Isolation Forest (using anomaly scores)
    iso_scores = iso.decision_function(X_test)
    PrecisionRecallDisplay.from_predictions(y_test, iso_scores, ax=ax, name="Isolation Forest")

    # 4. Final Touches
    ax.set_title("Model Comparison: Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()