import pandas as pd
import joblib
import os

def load_data(file):
    return pd.read_csv(file)

def save_model(model):
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/best_model.pkl")
