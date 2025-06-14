import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data_processing import DataProcessor
from models import ModelTrainer
from visualization import Visualizer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'suicide_risk_mapping_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)


def main():
    # Initialize components
    data_processor = DataProcessor(
        cdc_data_path='Mapping_Injury__Overdose__and_Violence_-_Census_Tract.csv',
        census_api_key=os.getenv('CENSUS_API_KEY')
    )
    model_trainer = ModelTrainer()
    visualizer = Visualizer()

    try:
        # Load and process data
        logging.info("Loading CDC data...")
        cdc_data = data_processor.load_cdc_data()

        logging.info("Fetching ACS data...")
        acs_data = data_processor.get_acs_data()

        logging.info("Fetching SVI data...")
        svi_data = data_processor.get_svi_data()

        logging.info("Creating spatial features...")
        spatial_data = data_processor.create_spatial_features(cdc_data)

        # Prepare features and labels
        logging.info("Preparing features and labels...")
        X, y = data_processor.prepare_features(cdc_data, acs_data, svi_data)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train models
        logging.info("Training models...")
        lr_results = model_trainer.train_lr(X_train, y_train)
        gbt_results = model_trainer.train_gbt(X_train, y_train)

        # Prepare GNN data
        logging.info("Preparing GNN data...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        # Create adjacency matrix (simplified - in practice, use actual tract adjacency)
        n = len(X_train)
        edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).t()

        # Train GNN
        logging.info("Training GNN...")
        gnn_results = model_trainer.train_gnn(
            X_train_tensor,
            edge_index,
            y_train_tensor,
            input_dim=X_train.shape[1]
        )

        # Evaluate models
        logging.info("Evaluating models...")
        evaluation_results = model_trainer.evaluate_models(
            X_test,
            y_test,
            X_test_gnn=torch.FloatTensor(X_test),
            edge_index_test=edge_index
        )

        # Create visualizations
        logging.info("Creating visualizations...")
        risk_map = visualizer.create_risk_map(
            spatial_data,
            model_trainer.gbt_model.predict_proba(X)[:, 1],  # Using GBT predictions
            cdc_data['GEOID'].values
        )

        # Create and run dashboard
        logging.info("Starting dashboard...")
        visualizer.create_dashboard(
            spatial_data,
            evaluation_results,
            {
                'gbt': model_trainer.gbt_model.predict_proba(X)[:, 1],
                'lr': model_trainer.lr_model.predict_proba(X)[:, 1],
                'gnn': model_trainer.gnn_model(X_train_tensor, edge_index).detach().numpy()
            }
        )
        visualizer.run_server()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
