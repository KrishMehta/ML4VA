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
from sklearn.neighbors import NearestNeighbors

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

        # Prepare features and labels
        logging.info("Preparing features and labels...")
        X, y, geoids = data_processor.prepare_features(cdc_data, acs_data, svi_data)
        
        # Create spatial features based on the GEOIDs we have data for
        logging.info("Creating spatial features...")
        tract_df = pd.DataFrame({'GEOID': geoids})
        spatial_data = data_processor.create_spatial_features(tract_df)

        # Debug: Check data shapes and statistics
        logging.info(f"Feature matrix shape: {X.shape}")
        logging.info(f"Label distribution - Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
        logging.info(f"Feature statistics - Mean: {np.mean(X):.3f}, Std: {np.std(X):.3f}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train models
        logging.info("Training models...")
        lr_results = model_trainer.train_lr(X_train, y_train)
        logging.info(f"LR Training Results: AUROC={lr_results['auroc']:.3f}, Precision={lr_results['precision']:.3f}")
        
        gbt_results = model_trainer.train_gbt(X_train, y_train)
        logging.info(f"GBT Training Results: AUROC={gbt_results['auroc']:.3f}, Precision={gbt_results['precision']:.3f}")

        # Prepare GNN data for training
        logging.info("Preparing GNN training data...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        # Create k-nearest neighbors adjacency matrix for training
        n_train = len(X_train)
        k_neighbors = min(10, n_train - 1)  # Use 10 nearest neighbors or less
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 because it includes self
        nn.fit(X_train)
        _, indices = nn.kneighbors(X_train)
        
        edges = []
        for i in range(n_train):
            for j in indices[i, 1:]:  # Skip self (index 0)
                edges.append([i, j])
        edge_index_train = torch.tensor(edges, dtype=torch.long).t()
        logging.info(f"Created training graph with {edge_index_train.shape[1]} edges")

        # Train GNN
        logging.info("Training GNN...")
        gnn_results = model_trainer.train_gnn(
            X_train_tensor,
            edge_index_train,
            y_train_tensor,
            input_dim=X_train.shape[1]
        )

        # Evaluate models
        logging.info("Evaluating models...")
        # Prepare GNN data for evaluation (X_test)
        X_test_tensor = torch.FloatTensor(X_test)
        n_test = len(X_test)
        
        # Create k-nearest neighbors adjacency matrix for test
        k_neighbors_test = min(10, n_test - 1)
        nn_test = NearestNeighbors(n_neighbors=k_neighbors_test + 1)
        nn_test.fit(X_test)
        _, indices_test = nn_test.kneighbors(X_test)
        
        edges_test = []
        for i in range(n_test):
            for j in indices_test[i, 1:]:
                edges_test.append([i, j])
        edge_index_test = torch.tensor(edges_test, dtype=torch.long).t()

        evaluation_results = model_trainer.evaluate_models(
            X_test,
            y_test,
            X_test_gnn=X_test_tensor,
            edge_index_test=edge_index_test
        )
        
        # Log evaluation results
        for model_name, results in evaluation_results.items():
            logging.info(f"{model_name.upper()} Test Results: AUROC={results['auroc']:.3f}, "
                        f"Precision={results['precision']:.3f}, Recall={results['recall']:.3f}, "
                        f"F1={results['f1']:.3f}, Accuracy={results['accuracy']:.3f}")
        
        # Calculate predictions on the full dataset X for visualization purposes
        logging.info("Calculating full dataset predictions for visualization...")
        all_X_lr_predictions = model_trainer.lr_model.predict_proba(X)[:, 1]
        all_X_gbt_predictions = model_trainer.gbt_model.predict_proba(X)[:, 1]
        
        # Debug predictions
        logging.info(f"LR predictions - Min: {all_X_lr_predictions.min():.3f}, Max: {all_X_lr_predictions.max():.3f}, Mean: {all_X_lr_predictions.mean():.3f}")
        logging.info(f"GBT predictions - Min: {all_X_gbt_predictions.min():.3f}, Max: {all_X_gbt_predictions.max():.3f}, Mean: {all_X_gbt_predictions.mean():.3f}")
        
        # Prepare GNN data for full dataset (X)
        X_full_tensor = torch.FloatTensor(X)
        n_full = len(X)
        
        # Create k-nearest neighbors adjacency matrix for full dataset
        k_neighbors_full = min(10, n_full - 1)
        nn_full = NearestNeighbors(n_neighbors=k_neighbors_full + 1)
        nn_full.fit(X)
        _, indices_full = nn_full.kneighbors(X)
        
        edges_full = []
        for i in range(n_full):
            for j in indices_full[i, 1:]:
                edges_full.append([i, j])
        edge_index_full = torch.tensor(edges_full, dtype=torch.long).t()
        
        # Ensure GNN model is in eval mode before making predictions
        model_trainer.gnn_model.eval()
        with torch.no_grad():
            all_X_gnn_predictions = model_trainer.gnn_model(X_full_tensor, edge_index_full).detach().numpy().flatten()

        logging.info(f"GNN predictions - Min: {all_X_gnn_predictions.min():.3f}, Max: {all_X_gnn_predictions.max():.3f}, Mean: {all_X_gnn_predictions.mean():.3f}")

        # Create visualizations
        logging.info("Creating risk map...")
        risk_map_gdf = visualizer.create_risk_map(
            spatial_data.copy(), # Pass a copy to avoid modifying original spatial_data
            all_X_gbt_predictions,  # Using GBT predictions for the map as specified in the image
            geoids
        )

        # Prepare risk scores dictionary for dashboard distribution
        risk_scores_for_dashboard = {
            'gbt': all_X_gbt_predictions,
            'lr': all_X_lr_predictions,
            'gnn': all_X_gnn_predictions
        }
        
        # Save results for standalone dashboard
        logging.info("Saving results for dashboard...")
        import pickle
        import json
        
        with open('risk_map_gdf.pkl', 'wb') as f:
            pickle.dump(risk_map_gdf, f)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f)
        
        with open('risk_scores.pkl', 'wb') as f:
            pickle.dump(risk_scores_for_dashboard, f)
        
        logging.info("Results saved. You can run the dashboard separately with: python run_dashboard.py")

        # Create and run dashboard
        logging.info("Starting dashboard...")
        visualizer.create_dashboard(
            risk_map_gdf, # Pass the GeoDataFrame that already has risk scores
            evaluation_results,
            risk_scores_for_dashboard # Pass the dictionary with all model predictions on full X
        )
        visualizer.run_server()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        
        # If it's just the dashboard error, try to provide alternative access info
        if "run_server" in str(e) or "app.run" in str(e):
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Dashboard failed to start automatically.")
            logging.info("The analysis is complete. Results have been logged.")
            logging.info("Model performance:")
            if 'evaluation_results' in locals():
                for model_name, results in evaluation_results.items():
                    logging.info(f"  {model_name.upper()}: AUROC={results['auroc']:.3f}")
        else:
            # For other errors, still exit
            raise


if __name__ == "__main__":
    main()
