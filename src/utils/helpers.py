"""
Helper utilities for model calibration and analysis.

This module provides various utility functions for data processing,
mathematical calculations, and common operations used throughout the package.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from scipy.stats import norm
import pandas as pd


def calculate_forward_price(spot: float, risk_free_rate: float, 
                          time_to_maturity: float, dividend_yield: float = 0.0) -> float:
    """
    Calculate forward price for an asset.
    
    Args:
        spot: Current spot price
        risk_free_rate: Risk-free interest rate
        time_to_maturity: Time to maturity in years
        dividend_yield: Continuous dividend yield
        
    Returns:
        Forward price
    """
    return spot * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)


def black_scholes_price(spot: float, strike: float, time_to_maturity: float,
                       risk_free_rate: float, volatility: float, 
                       option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_maturity: Time to maturity in years
        risk_free_rate: Risk-free interest rate
        volatility: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    if time_to_maturity <= 0:
        # Handle expiry case
        if option_type.lower() == 'call':
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)
    
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    if option_type.lower() == 'call':
        price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    else:
        price = strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return price


def calculate_greeks(spot: float, strike: float, time_to_maturity: float,
                    risk_free_rate: float, volatility: float, 
                    option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate option Greeks.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_maturity: Time to maturity in years
        risk_free_rate: Risk-free interest rate
        volatility: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with Greeks
    """
    if time_to_maturity <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    
    sqrt_t = np.sqrt(time_to_maturity)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    
    # Common terms
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)
    discount = np.exp(-risk_free_rate * time_to_maturity)
    
    if option_type.lower() == 'call':
        delta = nd1
        rho = strike * time_to_maturity * discount * nd2
        theta = (-spot * npd1 * volatility / (2 * sqrt_t) 
                - risk_free_rate * strike * discount * nd2)
    else:
        delta = nd1 - 1
        rho = -strike * time_to_maturity * discount * norm.cdf(-d2)
        theta = (-spot * npd1 * volatility / (2 * sqrt_t) 
                + risk_free_rate * strike * discount * norm.cdf(-d2))
    
    gamma = npd1 / (spot * volatility * sqrt_t)
    vega = spot * npd1 * sqrt_t
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega / 100,  # Convert to percentage
        'theta': theta / 365,  # Convert to daily
        'rho': rho / 100  # Convert to percentage
    }


def interpolate_volatility(strikes: List[float], volatilities: List[float],
                          target_strikes: List[float], 
                          method: str = 'linear') -> List[float]:
    """
    Interpolate volatilities for target strikes.
    
    Args:
        strikes: Known strikes
        volatilities: Known volatilities
        target_strikes: Strikes to interpolate for
        method: Interpolation method ('linear', 'cubic')
        
    Returns:
        Interpolated volatilities
    """
    from scipy.interpolate import interp1d
    
    # Remove NaN values
    valid_indices = [i for i, vol in enumerate(volatilities) if not np.isnan(vol)]
    clean_strikes = [strikes[i] for i in valid_indices]
    clean_vols = [volatilities[i] for i in valid_indices]
    
    if len(clean_strikes) < 2:
        raise ValueError("Need at least 2 valid points for interpolation")
    
    # Create interpolation function
    if method == 'linear':
        interp_func = interp1d(clean_strikes, clean_vols, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
    elif method == 'cubic':
        if len(clean_strikes) < 4:
            # Fall back to linear if not enough points for cubic
            interp_func = interp1d(clean_strikes, clean_vols, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        else:
            interp_func = interp1d(clean_strikes, clean_vols, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return interp_func(target_strikes).tolist()


def calculate_moneyness(strikes: Union[List[float], np.ndarray], 
                       spot: float, 
                       moneyness_type: str = 'log') -> List[float]:
    """
    Calculate different types of moneyness.
    
    Args:
        strikes: Strike prices
        spot: Spot price
        moneyness_type: Type of moneyness ('log', 'simple', 'forward')
        
    Returns:
        List of moneyness values
    """
    strikes = np.array(strikes)
    
    if moneyness_type == 'log':
        return (np.log(strikes / spot)).tolist()
    elif moneyness_type == 'simple':
        return (strikes / spot).tolist()
    elif moneyness_type == 'forward':
        return (strikes / spot - 1).tolist()
    else:
        raise ValueError(f"Unknown moneyness type: {moneyness_type}")


def validate_parameters(parameters: Dict[str, float], 
                       parameter_bounds: Dict[str, Tuple[float, float]]) -> List[str]:
    """
    Validate parameters against bounds.
    
    Args:
        parameters: Dictionary of parameter values
        parameter_bounds: Dictionary of parameter bounds
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for param_name, value in parameters.items():
        if param_name in parameter_bounds:
            lower, upper = parameter_bounds[param_name]
            if value < lower:
                errors.append(f"{param_name} = {value:.6f} is below lower bound {lower}")
            elif value > upper:
                errors.append(f"{param_name} = {value:.6f} is above upper bound {upper}")
    
    return errors


def calculate_calibration_metrics(predicted: np.ndarray, actual: np.ndarray,
                                weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive calibration metrics.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        weights: Optional weights
        
    Returns:
        Dictionary of metrics
    """
    if len(predicted) != len(actual):
        raise ValueError("Predicted and actual arrays must have same length")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predicted) | np.isnan(actual))
    if not np.any(valid_mask):
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r_squared': np.nan}
    
    pred_clean = predicted[valid_mask]
    actual_clean = actual[valid_mask]
    weights_clean = weights[valid_mask] if weights is not None else None
    
    # Calculate errors
    errors = pred_clean - actual_clean
    abs_errors = np.abs(errors)
    
    # Apply weights if provided
    if weights_clean is not None:
        errors = errors * weights_clean
        abs_errors = abs_errors * weights_clean
        actual_weighted = actual_clean * weights_clean
        pred_weighted = pred_clean * weights_clean
    else:
        actual_weighted = actual_clean
        pred_weighted = pred_clean
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(abs_errors)
    
    # MAPE
    nonzero_mask = actual_clean != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((pred_clean[nonzero_mask] - actual_clean[nonzero_mask]) / 
                             actual_clean[nonzero_mask]) * 100)
    else:
        mape = np.nan
    
    # R-squared
    if weights_clean is not None:
        actual_mean = np.average(actual_clean, weights=weights_clean)
        ss_tot = np.sum(weights_clean * (actual_clean - actual_mean)**2)
        ss_res = np.sum(weights_clean * errors**2)
    else:
        actual_mean = np.mean(actual_clean)
        ss_tot = np.sum((actual_clean - actual_mean)**2)
        ss_res = np.sum(errors**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r_squared': r_squared,
        'num_points': len(pred_clean)
    }


def generate_strike_grid(spot: float, min_moneyness: float = -0.5, 
                        max_moneyness: float = 0.5, num_points: int = 21) -> List[float]:
    """
    Generate a grid of strikes based on moneyness range.
    
    Args:
        spot: Spot price
        min_moneyness: Minimum log-moneyness
        max_moneyness: Maximum log-moneyness
        num_points: Number of strikes to generate
        
    Returns:
        List of strike prices
    """
    log_moneyness_grid = np.linspace(min_moneyness, max_moneyness, num_points)
    strikes = spot * np.exp(log_moneyness_grid)
    return strikes.tolist()


def format_parameter_summary(parameters: Dict[str, float], 
                           precision: int = 6) -> str:
    """
    Format parameter summary for display.
    
    Args:
        parameters: Dictionary of parameters
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in parameters.items():
        lines.append(f"{name:>10}: {value:>{precision+8}.{precision}f}")
    
    return "\n".join(lines)


def create_correlation_matrix(data: pd.DataFrame, 
                            method: str = 'pearson') -> pd.DataFrame:
    """
    Create correlation matrix for analysis.
    
    Args:
        data: DataFrame with data
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    return data.corr(method=method)
