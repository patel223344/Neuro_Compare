import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # Random Forest ki jagah Logistic Regression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_comparison():
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Traditional ML (Logistic Regression - 100% safe from DLL issues)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    # 2. Neural Network (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    
    results = {
        "Traditional_ML_RandomForest": {
            "Accuracy": round(accuracy_score(y_test, lr_pred) * 100, 2),
            "Precision": round(precision_score(y_test, lr_pred) * 100, 2),
            "Recall": round(recall_score(y_test, lr_pred) * 100, 2),
            "F1_Score": round(f1_score(y_test, lr_pred) * 100, 2)
        },
        "Neural_Network_MLP": {
            "Accuracy": round(accuracy_score(y_test, mlp_pred) * 100, 2),
            "Precision": round(precision_score(y_test, mlp_pred) * 100, 2),
            "Recall": round(recall_score(y_test, mlp_pred) * 100, 2),
            "F1_Score": round(f1_score(y_test, mlp_pred) * 100, 2)
        }
    }
    
    return results