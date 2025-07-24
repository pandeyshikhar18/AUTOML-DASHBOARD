import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import LabelEncoder

def train_and_select(df, target_col, model_path="model.pkl"):
    # Split features & target
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Drop non‑informative columns
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            X.drop(columns=[col], inplace=True)

    # Fill missing values
    X.fillna(method="ffill", inplace=True)
    X.fillna(method="bfill", inplace=True)
    X.fillna(0, inplace=True)
    if y.dtype == object:
        y.fillna(y.mode()[0], inplace=True)
    else:
        y.fillna(y.median(), inplace=True)

    # Detect classification vs regression
    classification = (y.dtype == object) or (y.nunique() <= 10)
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # One‑hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    try:
        strat = y if classification else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Candidate models
    if classification:
        candidates = [
            ("Random Forest", RandomForestClassifier()),
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Decision Tree", DecisionTreeClassifier()),
            ("SVM", SVC(probability=True)),
            ("Naive Bayes", GaussianNB()),
        ]
        score_fn = accuracy_score
    else:
        candidates = [
            ("Random Forest Regressor", RandomForestRegressor()),
            ("Linear Regression", LinearRegression()),
            ("Decision Tree Regressor", DecisionTreeRegressor()),
            ("SVR", SVR()),
        ]
        score_fn = r2_score

    best_score = -np.inf
    best_model = None
    best_name = ""

    # Try each model
    for name, model in candidates:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            sc = score_fn(y_test, preds)
            if sc > best_score:
                best_score, best_model, best_name = sc, model, name
        except:
            continue

    # Fallback if none succeeded
    if best_model is None:
        if classification:
            fallback = DummyClassifier(strategy="most_frequent")
        else:
            fallback = DummyRegressor(strategy="mean")
        fallback.fit(X_train, y_train)
        preds = fallback.predict(X_test)
        best_score = score_fn(y_test, preds)
        best_model, best_name = fallback, fallback.__class__.__name__

    # Save and return
    joblib.dump((best_model, X_train), model_path)
    return best_model, best_name, best_score, X_train, y_train, classification
