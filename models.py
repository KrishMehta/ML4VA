import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from torch_geometric.nn import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
from typing import Tuple, Dict, Any


class GNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, num_heads: int = 4):
        super(GNNModel, self).__init__()
        # First layer: GAT with multiple attention heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        # Second layer: GCN for spatial smoothing
        self.conv2 = GCNConv(hidden_dim * num_heads, hidden_dim)
        # Final layer: MLP for prediction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First layer with attention
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        # Second layer with spatial smoothing
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # Final MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)


class ModelTrainer:
    def __init__(self):
        self.lr_model = LogisticRegression(max_iter=1000)
        self.gbt_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.gnn_model = None

    def train_lr(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train logistic regression model."""
        self.lr_model.fit(X_train, y_train)
        y_proba = self.lr_model.predict_proba(X_train)[:, 1]
        return self._calculate_metrics(y_train, y_proba)

    def train_gbt(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train gradient boosted trees model."""
        self.gbt_model.fit(X_train, y_train)
        y_proba = self.gbt_model.predict_proba(X_train)[:, 1]
        return self._calculate_metrics(y_train, y_proba)

    def train_gnn(self,
                  X_train: torch.Tensor,
                  edge_index: torch.Tensor,
                  y_train: torch.Tensor,
                  input_dim: int,
                  hidden_dim: int = 64,
                  num_epochs: int = 200) -> Dict[str, float]:
        """Train graph neural network model."""
        self.gnn_model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

        self.gnn_model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self.gnn_model(X_train, edge_index)
            loss = F.binary_cross_entropy(out, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

        # Evaluate
        self.gnn_model.eval()
        with torch.no_grad():
            y_proba = self.gnn_model(X_train, edge_index).numpy()

        return self._calculate_metrics(y_train.numpy(), y_proba)

    def _calculate_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        # Ensure 1-D array of probabilities
        y_proba = np.asarray(y_proba).reshape(-1)
        # Guard against probabilities outside [0,1]
        y_proba = np.clip(y_proba, 0, 1)

        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate calibration curve
        # Some bins may be empty which causes warnings; suppress for now
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
        
        return {
            'auroc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
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
        
        # Calculate metrics for each model
        results = {
            'lr': self._calculate_metrics(y_test, y_proba_lr),
            'gbt': self._calculate_metrics(y_test, y_proba_gbt),
            'gnn': self._calculate_metrics(y_test, y_proba_gnn)
        }
        
        return results
