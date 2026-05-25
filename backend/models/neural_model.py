import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class NeuralModel:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.learning_curve = []

    def train(self, X_train, y_train, epochs=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Split out a small validation set from train
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        self.learning_curve = []
        self.model.train()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            outputs = self.model(X_tr_t)
            loss = criterion(outputs, y_tr_t)
            loss.backward()
            optimizer.step()

            # Compute val loss
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            self.model.train()

            self.learning_curve.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
            })

        self.model.eval()
        return self.learning_curve

    def predict(self, X):
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            probs = self.model(X_t).squeeze(1).numpy()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            probs = self.model(X_t).squeeze(1).numpy()
        return np.column_stack([1 - probs, probs])

    def get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
        }
