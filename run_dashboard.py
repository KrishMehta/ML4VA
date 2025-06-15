#!/usr/bin/env python3
"""
Standalone script to run the Virginia Suicide Risk Analysis Dashboard.
This script loads pre-computed results and displays them in an interactive dashboard.
"""

import pickle
import json
from visualization import Visualizer
import logging
import sys

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Load pre-computed results
        with open('risk_map_gdf.pkl', 'rb') as f:
            risk_map_gdf = pickle.load(f)
        
        with open('evaluation_results.json', 'r') as f:
            evaluation_results = json.load(f)
        
        with open('risk_scores.pkl', 'rb') as f:
            risk_scores = pickle.load(f)
        
        # Create visualizer and dashboard
        visualizer = Visualizer()
        visualizer.create_dashboard(risk_map_gdf, evaluation_results, risk_scores)
        
        logging.info("Starting dashboard on http://localhost:8050")
        logging.info("Press Ctrl+C to stop the server")
        
        # Run the server
        visualizer.app.run(debug=False, host='0.0.0.0', port=8050)
        
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        logging.error("Please run main.py first to generate the analysis results.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 