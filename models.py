import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
from typing import Tuple, Dict, Any


class GNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x)


class ModelTrainer:
    def __init__(self):
        self.lr_model = LogisticRegression()
        self.gbt_model = GradientBoostingClassifier(random_state=42)
        self.gnn_model = None

    def train_lr(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train logistic regression model."""
        self.lr_model.fit(X_train, y_train)
        y_proba = self.lr_model.predict_proba(X_train)[:, 1]
        return {
            'auroc': roc_auc_score(y_train, y_proba),
            'precision': precision_score(y_train, (y_proba >= 0.5).astype(int)),
            'recall': recall_score(y_train, (y_proba >= 0.5).astype(int))
        }

    def train_gbt(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train gradient boosted trees model."""
        self.gbt_model.fit(X_train, y_train)
        y_proba = self.gbt_model.predict_proba(X_train)[:, 1]
        return {
            'auroc': roc_auc_score(y_train, y_proba),
            'precision': precision_score(y_train, (y_proba >= 0.5).astype(int)),
            'recall': recall_score(y_train, (y_proba >= 0.5).astype(int))
        }

    def train_gnn(self,
                  X_train: torch.Tensor,
                  edge_index: torch.Tensor,
                  y_train: torch.Tensor,
                  input_dim: int,
                  hidden_dim: int = 64) -> Dict[str, float]:
        """Train graph neural network model."""
        self.gnn_model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)

        self.gnn_model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = self.gnn_model(X_train, edge_index)
            loss = F.binary_cross_entropy(out, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

        # Evaluate
        self.gnn_model.eval()
        with torch.no_grad():
            y_proba = self.gnn_model(X_train, edge_index).numpy()

        return {
            'auroc': roc_auc_score(y_train.numpy(), y_proba),
            'precision': precision_score(y_train.numpy(), (y_proba >= 0.5).astype(int)),
            'recall': recall_score(y_train.numpy(), (y_proba >= 0.5).astype(int))
        }

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, X_test_gnn: torch.Tensor, edge_index_test: torch.Tensor) -> Dict:
        """Evaluate all models on test data."""
        # Convert test data to tensors
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Get predictions
        y_pred_lr = self.lr_model.predict(X_test)
        y_pred_gbt = self.gbt_model.predict(X_test)
        y_proba_lr = self.lr_model.predict_proba(X_test)[:, 1]
        y_proba_gbt = self.gbt_model.predict_proba(X_test)[:, 1]
        
        # Ensure edge_index_test is within bounds
        n_nodes = X_test_gnn.size(0)
        edge_index_test = edge_index_test[:, edge_index_test[0] < n_nodes]
        edge_index_test = edge_index_test[:, edge_index_test[1] < n_nodes]
        
        # Get GNN predictions
        with torch.no_grad():
            y_proba_gnn = self.gnn_model(X_test_gnn, edge_index_test).numpy()
        y_pred_gnn = (y_proba_gnn > 0.5).astype(int)
        
        # Calculate metrics
        results = {
            'lr': {
                'accuracy': accuracy_score(y_test, y_pred_lr),
                'precision': precision_score(y_test, y_pred_lr, zero_division=0),
                'recall': recall_score(y_test, y_pred_lr, zero_division=0),
                'f1': f1_score(y_test, y_pred_lr, zero_division=0),
                'auroc': roc_auc_score(y_test, y_proba_lr)
            },
            'gbt': {
                'accuracy': accuracy_score(y_test, y_pred_gbt),
                'precision': precision_score(y_test, y_pred_gbt, zero_division=0),
                'recall': recall_score(y_test, y_pred_gbt, zero_division=0),
                'f1': f1_score(y_test, y_pred_gbt, zero_division=0),
                'auroc': roc_auc_score(y_test, y_proba_gbt)
            },
            'gnn': {
                'accuracy': accuracy_score(y_test, y_pred_gnn),
                'precision': precision_score(y_test, y_pred_gnn, zero_division=0),
                'recall': recall_score(y_test, y_pred_gnn, zero_division=0),
                'f1': f1_score(y_test, y_pred_gnn, zero_division=0),
                'auroc': roc_auc_score(y_test, y_proba_gnn)
            }
        }
        
        return results
