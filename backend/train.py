import time
import json
import os
import joblib
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from feature_engineering import load_and_prepare
from models.logistic_model import LogisticModel
from models.forest_model import ForestModel
from models.neural_model import NeuralModel
from evaluate import compute_metrics


def train_all():
    os.makedirs('saved_models', exist_ok=True)

    # Load and prepare data
    X_scaled, y, feature_names = load_and_prepare()

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    metrics_all = {}

    # --- Logistic Regression ---
    logistic = LogisticModel()
    t0 = time.time()
    logistic.train(X_train, y_train)
    logistic_train_time = time.time() - t0

    logistic_metrics = compute_metrics(logistic, X_test, y_test)
    logistic_metrics['training_time'] = logistic_train_time
    metrics_all['logistic'] = logistic_metrics
    print(f"Logistic Regression accuracy: {logistic_metrics['accuracy']:.4f}")

    joblib.dump(logistic, 'saved_models/logistic_model.joblib')

    # --- Random Forest ---
    forest = ForestModel()
    t0 = time.time()
    forest.train(X_train, y_train, feature_names=feature_names)
    forest_train_time = time.time() - t0

    forest_metrics = compute_metrics(forest, X_test, y_test)
    forest_metrics['training_time'] = forest_train_time
    # Store feature importance in metrics
    forest_metrics['feature_importance'] = [
        {"name": name, "importance": float(imp)}
        for name, imp in forest.get_feature_importance()
    ]
    metrics_all['forest'] = forest_metrics
    print(f"Random Forest accuracy: {forest_metrics['accuracy']:.4f}")

    joblib.dump(forest, 'saved_models/forest_model.joblib')

    # --- Neural Network ---
    neural = NeuralModel()
    t0 = time.time()
    learning_curve = neural.train(X_train, y_train, epochs=50)
    neural_train_time = time.time() - t0

    neural_metrics = compute_metrics(neural, X_test, y_test)
    neural_metrics['training_time'] = neural_train_time
    neural_metrics['learning_curve'] = learning_curve
    metrics_all['neural'] = neural_metrics
    print(f"Neural Network accuracy: {neural_metrics['accuracy']:.4f}")

    torch.save(neural, 'saved_models/neural_model.pt')

    # Save all metrics to JSON
    with open('saved_models/metrics.json', 'w') as f:
        json.dump(metrics_all, f, indent=2)

    print("Training complete. Metrics saved to saved_models/metrics.json")
    return metrics_all


if __name__ == '__main__':
    train_all()
