import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import numpy as np
from typing import Dict, List, Tuple
import dash
import logging


class Visualizer:
    def __init__(self):
        self.app = Dash(__name__)

    def create_risk_map(self,
                        gdf: gpd.GeoDataFrame,
                        risk_scores: np.ndarray,
                        tract_ids: np.ndarray) -> gpd.GeoDataFrame:
        """Create a risk map using the predicted risk scores."""
        logging.info(f"[create_risk_map] Input GDF shape: {gdf.shape}")
        logging.info(f"[create_risk_map] Risk scores shape: {risk_scores.shape}")
        logging.info(f"[create_risk_map] Tract IDs shape: {tract_ids.shape}")
        
        # Create a DataFrame with risk scores and tract IDs
        risk_df = pd.DataFrame({
            'GEOID': tract_ids,
            'risk_score': risk_scores
        })
        
        # Group by GEOID and take the mean risk score for each tract
        risk_df = risk_df.groupby('GEOID')['risk_score'].mean().reset_index()
        logging.info(f"[create_risk_map] Number of unique tracts with risk scores: {len(risk_df)}")
        
        # Add risk scores to GeoDataFrame
        # Keep only necessary columns to avoid conflicts
        gdf_clean = gdf[['GEOID', 'geometry']].drop_duplicates(subset=['GEOID'])
        gdf_merged = gdf_clean.merge(risk_df, on='GEOID', how='left')
        
        # Fill any missing risk scores with 0
        gdf_merged['risk_score'] = gdf_merged['risk_score'].fillna(0)
        
        logging.info(f"[create_risk_map] Output GDF shape: {gdf_merged.shape}")
        logging.info(f"[create_risk_map] Risk score range: [{gdf_merged['risk_score'].min():.3f}, {gdf_merged['risk_score'].max():.3f}]")
        
        return gdf_merged

    def create_dashboard(self, gdf: gpd.GeoDataFrame, evaluation_results: Dict, risk_scores: Dict[str, np.ndarray]) -> None:
        """Create and run the dashboard."""
        # Use the Dash app initialized in __init__
        app = self.app

        # Build the layout
        app.layout = html.Div([
            html.H1("Virginia Suicide Risk Analysis Dashboard", 
                   style={'textAlign': 'center', 'margin': '20px'}),
            
            # Map visualization
            html.Div([
                html.H2("Risk Map", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='risk-map',
                    figure=self.create_map_figure(
                        gdf
                    )
                )
            ], style={'width': '100%', 'margin': '20px'}),
            
            # Model performance metrics
            html.Div([
                html.H2("Model Performance", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='performance-metrics',
                    figure=self.create_performance_figure(evaluation_results)
                )
            ], style={'width': '100%', 'margin': '20px'}),
            
            # Risk score distribution
            html.Div([
                html.H2("Risk Score Distribution", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='risk-distribution',
                    figure=self.create_distribution_figure(risk_scores)
                )
            ], style={'width': '100%', 'margin': '20px'})
        ])
        
        # Layout is now set; caller can start server via `run_server()`.

    def create_map_figure(self, gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create a choropleth map figure."""
        # Ensure we have the required columns
        if 'risk_score' not in gdf.columns:
            raise ValueError("GeoDataFrame must contain 'risk_score' column")

        # Plotly expects geometries in WGS84 (EPSG:4326). Re-project if necessary.
        try:
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        except Exception as crs_err:
            # Log or print CRS issues but continue; the map may still render if already correct.
            logging.warning(f"[Visualizer] CRS conversion warning: {crs_err}")
        
        # Add a NAME column if it doesn't exist (for hover display)
        if 'NAME' not in gdf.columns:
            gdf['NAME'] = gdf['GEOID']

        # Create the choropleth map
        gdf_json = json.loads(gdf.to_json())
        fig = px.choropleth(
            gdf,
            geojson=gdf_json,
            locations='GEOID',
            featureidkey='properties.GEOID',
            color='risk_score',
            color_continuous_scale='YlOrRd',
            range_color=(0, 1),
            labels={'risk_score': 'Risk Score'},
            title='Virginia Suicide Risk Map'
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>Tract %{properties.GEOID}</b><br>Risk Score: %{z:.3f}<extra></extra>'
        )
        
        # Update the layout
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            bgcolor="white"
        )
        
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            title_x=0.5,
            height=600,  # Set a fixed height
            width=1000,   # Set a fixed width
            geo=dict(
                bgcolor="rgba(0,0,0,0)"
            )
        )
        
        return fig

    def create_performance_figure(self, evaluation_results: Dict) -> go.Figure:
        """Create a bar chart of model performance metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
        models = list(evaluation_results.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [evaluation_results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis_range=[0, 1]
        )
        
        return fig

    def create_distribution_figure(self, risk_scores: Dict[str, np.ndarray]) -> go.Figure:
        """Create a histogram of risk score distributions."""
        fig = go.Figure()
        
        for model_name, scores in risk_scores.items():
            fig.add_trace(go.Histogram(
                x=scores,
                name=model_name,
                opacity=0.7
            ))
        
        fig.update_layout(
            title='Risk Score Distribution by Model',
            xaxis_title='Risk Score',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig

    def _create_metrics_plot(self, model_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a plot of model performance metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
        models = list(model_results.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis_range=[0, 1]
        )
        
        return fig

    def _create_risk_map_plot(self,
                              gdf: gpd.GeoDataFrame,
                              risk_scores: Dict[str, np.ndarray]) -> go.Figure:
        """Create an interactive risk map using Plotly."""
        fig = px.choropleth(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color='risk_score',
            color_continuous_scale='YlOrRd',
            title='Suicide Risk by Census Tract'
        )

        fig.update_geos(
            fitbounds="locations",
            visible=False
        )

        return fig

    def _create_feature_importance_plot(self) -> go.Figure:
        """Create a bar plot of feature importance."""
        # This would be implemented based on your model's feature importance
        # For now, returning a placeholder
        return go.Figure()

    def run_server(self, **kwargs):
        """Run the Dash server. Accepts the same kwargs as Dash.run_server."""
        # Provide sensible defaults but allow overrides
        params = {
            'debug': False,
            'port': 8050,
            'host': '0.0.0.0'
        }
        params.update(kwargs)
        
        # Try run_server first, then fall back to run if it doesn't exist
        try:
            self.app.run_server(**params)
        except Exception as e:
            # In Dash 3.0, run_server was replaced by run
            if "run_server" in str(e) or "ObsoleteAttributeException" in str(type(e)):
                self.app.run(**params)
            else:
                raise
        except AttributeError:
            # Fallback for other potential attribute errors
            self.app.run(**params)
