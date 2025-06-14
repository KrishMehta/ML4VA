import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import numpy as np
from typing import Dict, List, Tuple
import dash


class Visualizer:
    def __init__(self):
        self.app = Dash(__name__)

    def create_risk_map(self,
                        gdf: gpd.GeoDataFrame,
                        risk_scores: np.ndarray,
                        tract_ids: np.ndarray) -> gpd.GeoDataFrame:
        """Create a risk map using the predicted risk scores."""
        # Create a DataFrame with risk scores and tract IDs
        risk_df = pd.DataFrame({
            'GEOID': tract_ids,
            'risk_score': risk_scores
        })
        
        # Group by GEOID and take the mean risk score for each tract
        risk_df = risk_df.groupby('GEOID')['risk_score'].mean().reset_index()
        print(f"Number of unique tracts: {len(risk_df)}")
        
        # Add risk scores to GeoDataFrame
        gdf = gdf.merge(risk_df, on='GEOID', how='left')
        
        # Fill any missing risk scores with 0
        gdf['risk_score'] = gdf['risk_score'].fillna(0)
        
        return gdf

    def create_dashboard(self, gdf: gpd.GeoDataFrame, evaluation_results: Dict, risk_scores: Dict[str, np.ndarray]) -> None:
        """Create and run the dashboard."""
        # Initialize the Dash app
        app = dash.Dash(__name__)
        
        # Create the layout
        app.layout = html.Div([
            html.H1("Virginia Suicide Risk Analysis Dashboard", 
                   style={'textAlign': 'center', 'margin': '20px'}),
            
            # Map visualization
            html.Div([
                html.H2("Risk Map", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='risk-map',
                    figure=self.create_map_figure(
                        self.create_risk_map(gdf, risk_scores['gbt'], gdf['GEOID'].values)
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
        
        # Run the server
        app.run(debug=True, port=8050)

    def create_map_figure(self, gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create a choropleth map figure."""
        # Ensure we have the required columns
        if 'risk_score' not in gdf.columns:
            raise ValueError("GeoDataFrame must contain 'risk_score' column")
        
        # Create the choropleth map
        fig = px.choropleth(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color='risk_score',
            color_continuous_scale='YlOrRd',
            range_color=(0, 1),
            labels={'risk_score': 'Risk Score'},
            title='Virginia Suicide Risk Map',
            hover_data={
                'GEOID': True,
                'NAME': True,
                'risk_score': ':.3f'
            }
        )
        
        # Update the layout
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            center=dict(lat=37.5215, lon=-78.6569),  # Center on Virginia
            projection_scale=7  # Adjust zoom level
        )
        
        fig.update_layout(
            margin={"r":0,"t":30,"l":0,"b":0},
            title_x=0.5,
            height=600,  # Set a fixed height
            width=800    # Set a fixed width
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
