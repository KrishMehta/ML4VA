# Virginia Suicide Risk Mapping Tool

This project implements a tract-level suicide risk mapping tool for Virginia by integrating CDC mortality data with socio-demographic features from the American Community Survey (ACS) and Social Vulnerability Index (SVI) data.

## Features

- Data integration from multiple sources (CDC, ACS, SVI)
- Multiple modeling approaches:
  - Logistic Regression
  - Gradient-Boosted Trees
  - Graph Neural Network with spatial dependencies
- Interactive dashboard with:
  - Risk maps
  - Model performance metrics
  - Feature importance visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/KrishMehta/ML4VA
cd ML4VA
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export CENSUS_API_KEY='your-census-api-key'
```

## Data Requirements

1. CDC Mortality Data:
   - File: `Mapping_Injury__Overdose__and_Violence_-_Census_Tract.csv`

2. Census API Key:
   - Required for accessing ACS data
   - Get from: https://api.census.gov/data/key_signup.html

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Load and process all data sources
2. Train the models
3. Generate visualizations
4. Launch the interactive dashboard at http://localhost:8050

## Project Structure

- `main.py`: Main script orchestrating the pipeline
- `data_processing.py`: Data loading and preprocessing
- `models.py`: Model implementations and training
- `visualization.py`: Dashboard and visualization components
- `requirements.txt`: Project dependencies

## Model Performance

The current implementation includes three models:
1. Logistic Regression (baseline)
2. Gradient-Boosted Trees
3. Graph Neural Network

Performance metrics (AUROC, precision, recall) are displayed in the dashboard.

## Acknowledgments

- CDC WONDER Database for mortality data
- US Census Bureau for ACS data
- CDC/ATSDR for Social Vulnerability Index data 