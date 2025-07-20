"""
Plotting utilities for volatility surfaces and model analysis.

This module provides comprehensive plotting functionality for visualizing
option data, volatility surfaces, and model calibration results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict, Any, Tuple, Union
import pandas as pd

from ..data.option_data import OptionData
from ..data.market_data import MarketData


class VolatilitySurfacePlotter:
    """
    Plotter for volatility surfaces and related visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'default'):
        """
        Initialize the plotter.

        Args:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            # Fallback if style is not available
            plt.style.use('default')
    
    def plot_implied_volatility_smile(self, option_data: OptionData, 
                                    model_predictions: Optional[np.ndarray] = None,
                                    model_name: str = "Model",
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot implied volatility smile.
        
        Args:
            option_data: Option data to plot
            model_predictions: Optional model predictions to overlay
            model_name: Name of the model for legend
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot market data
        valid_indices = [i for i, iv in enumerate(option_data.implied_volatilities) 
                        if not np.isnan(iv)]
        
        market_moneyness = [option_data.log_moneyness[i] for i in valid_indices]
        market_ivs = [option_data.implied_volatilities[i] for i in valid_indices]
        
        ax.scatter(market_moneyness, market_ivs, color='blue', alpha=0.7, 
                  s=50, label='Market Data', zorder=3)
        
        # Plot model predictions if provided
        if model_predictions is not None:
            model_moneyness = [option_data.log_moneyness[i] for i in valid_indices]
            ax.plot(model_moneyness, model_predictions, color='red', linewidth=2,
                   label=f'{model_name} Fit', zorder=2)
        
        ax.set_xlabel('Log-Moneyness ln(K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(title or f'Implied Volatility Smile (T={option_data.time_to_maturity:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volatility_surface_3d(self, market_data: MarketData,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D volatility surface.
        
        Args:
            market_data: Market data with multiple expiries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Collect data for surface
        moneyness_data = []
        time_data = []
        iv_data = []
        
        for expiry, option_data in market_data.option_chains.items():
            for i, iv in enumerate(option_data.implied_volatilities):
                if not np.isnan(iv):
                    moneyness_data.append(option_data.log_moneyness[i])
                    time_data.append(option_data.time_to_maturity)
                    iv_data.append(iv)
        
        if not moneyness_data:
            raise ValueError("No valid data points for surface plot")
        
        # Create scatter plot
        scatter = ax.scatter(moneyness_data, time_data, iv_data, 
                           c=iv_data, cmap=cm.viridis, s=50, alpha=0.8)
        
        ax.set_xlabel('Log-Moneyness ln(K/S)')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(title or f'Volatility Surface - {market_data.underlying_symbol}')
        
        # Add colorbar
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Implied Volatility')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volatility_surface_heatmap(self, market_data: MarketData,
                                      title: Optional[str] = None,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot volatility surface as a heatmap.
        
        Args:
            market_data: Market data with multiple expiries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get volatility surface DataFrame
        surface_df = market_data.get_volatility_surface()
        
        if surface_df.empty:
            raise ValueError("No data available for heatmap")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(surface_df.values, cmap='viridis', aspect='auto', 
                      interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(surface_df.columns)))
        ax.set_xticklabels(surface_df.columns, rotation=45)
        ax.set_yticks(range(len(surface_df.index)))
        ax.set_yticklabels([f'{strike:.0f}' for strike in surface_df.index])
        
        ax.set_xlabel('Expiry')
        ax.set_ylabel('Strike')
        ax.set_title(title or f'Volatility Surface Heatmap - {market_data.underlying_symbol}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Implied Volatility')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, option_data: OptionData,
                            model_results: Dict[str, np.ndarray],
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple model fits.
        
        Args:
            option_data: Option data
            model_results: Dictionary mapping model names to predictions
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot market data
        valid_indices = [i for i, iv in enumerate(option_data.implied_volatilities) 
                        if not np.isnan(iv)]
        
        market_moneyness = [option_data.log_moneyness[i] for i in valid_indices]
        market_ivs = [option_data.implied_volatilities[i] for i in valid_indices]
        
        ax.scatter(market_moneyness, market_ivs, color='black', alpha=0.8, 
                  s=60, label='Market Data', zorder=3)
        
        # Plot model predictions
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))
        
        for (model_name, predictions), color in zip(model_results.items(), colors):
            ax.plot(market_moneyness, predictions, color=color, linewidth=2,
                   label=f'{model_name}', zorder=2)
        
        ax.set_xlabel('Log-Moneyness ln(K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(title or 'Model Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ImpliedVolatilityPlotter:
    """
    Specialized plotter for implied volatility analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the IV plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_iv_term_structure(self, market_data: MarketData, 
                              strike_level: str = 'atm',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot implied volatility term structure.
        
        Args:
            market_data: Market data with multiple expiries
            strike_level: Strike level to plot ('atm', 'otm_call', 'otm_put')
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        expiries = []
        times_to_maturity = []
        ivs = []
        
        for expiry, option_data in market_data.option_chains.items():
            # Select strike based on level
            if strike_level == 'atm':
                # Find closest to ATM
                atm_index = np.argmin([abs(lm) for lm in option_data.log_moneyness])
                iv = option_data.implied_volatilities[atm_index]
            elif strike_level == 'otm_call':
                # Find OTM call (positive moneyness)
                otm_indices = [i for i, lm in enumerate(option_data.log_moneyness) if lm > 0.1]
                if otm_indices:
                    iv = option_data.implied_volatilities[otm_indices[0]]
                else:
                    continue
            elif strike_level == 'otm_put':
                # Find OTM put (negative moneyness)
                otm_indices = [i for i, lm in enumerate(option_data.log_moneyness) if lm < -0.1]
                if otm_indices:
                    iv = option_data.implied_volatilities[otm_indices[-1]]
                else:
                    continue
            else:
                raise ValueError(f"Unknown strike level: {strike_level}")
            
            if not np.isnan(iv):
                expiries.append(expiry)
                times_to_maturity.append(option_data.time_to_maturity)
                ivs.append(iv)
        
        if not times_to_maturity:
            raise ValueError("No valid data for term structure")
        
        # Sort by time to maturity
        sorted_data = sorted(zip(times_to_maturity, ivs, expiries))
        times_to_maturity, ivs, expiries = zip(*sorted_data)
        
        ax.plot(times_to_maturity, ivs, 'o-', linewidth=2, markersize=8)
        
        ax.set_xlabel('Time to Maturity (years)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(title or f'IV Term Structure - {strike_level.upper()}')
        ax.grid(True, alpha=0.3)
        
        # Add expiry labels
        for t, iv, exp in zip(times_to_maturity, ivs, expiries):
            ax.annotate(exp, (t, iv), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(self, option_data: OptionData, model_predictions: np.ndarray,
                      model_name: str = "Model", title: Optional[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model residuals.
        
        Args:
            option_data: Option data
            model_predictions: Model predictions
            model_name: Name of the model
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate residuals
        valid_indices = [i for i, iv in enumerate(option_data.implied_volatilities) 
                        if not np.isnan(iv)]
        
        market_ivs = [option_data.implied_volatilities[i] for i in valid_indices]
        market_moneyness = [option_data.log_moneyness[i] for i in valid_indices]
        
        residuals = np.array(model_predictions) - np.array(market_ivs)
        
        # Plot 1: Residuals vs Moneyness
        ax1.scatter(market_moneyness, residuals, alpha=0.7, s=50)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Log-Moneyness ln(K/S)')
        ax1.set_ylabel('Residuals (Model - Market)')
        ax1.set_title(f'{model_name} Residuals vs Moneyness')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals histogram
        ax2.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{model_name} Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title or f'{model_name} Residual Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
