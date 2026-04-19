import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_model(svm_model, filename):
    """
    Serialize a model to a file
    :param svm_model: Model to save
    :param filename: Destination file
    :return: None
    """
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)

def load_model(filename):
    """
    Load a model from a file
    :param filename: File to load
    :return: Loaded model
    """
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        print(f"{filename} is empty or corrupted.")
        return None
    except FileNotFoundError:
        print(f"Model file not found: {filename}")
        return None

def output_timing(seconds):
    """
    Transforms seconds into a human-readable string
    :param seconds: Number of seconds
    :return: Human-readable string
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    times = [f"{int(hours)} hours" if hours > 0 else None, f"{int(minutes)} minutes" if minutes > 0 else None, f"{int(seconds)} seconds"]
    return ", ".join(time for time in times if time is not None)

def generate_confusion_matrix(y_test, y_pred, figure_path, title):
    y_pred = (y_pred + 1) // 2
    y_test = (y_test + 1) // 2
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({title})")
    plt.savefig(figure_path)
    print(f"Confusion matrix saved to {figure_path}")
    return cm