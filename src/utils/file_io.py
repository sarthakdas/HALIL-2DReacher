import json
import pickle
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_data(data, filepath):
    # create the folder if it does not exist
    folder_path = os.path.dirname(filepath)
    create_folder_if_not_exists(folder_path)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
