import pickle

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