"""
SVI (Stochastic Volatility Inspired) model implementation.

This module provides the SVI model for volatility surface modeling,
including calibration methods and parameter constraints.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Union
from ..data.option_data import OptionData


class SVI:
    """
    SVI (Stochastic Volatility Inspired) model for volatility surface modeling.
    
    The SVI model parameterizes the implied variance as a function of log-moneyness:
    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
    
    where:
    - w(k) is the implied variance (volatility²)
    - k is the log-moneyness ln(K/F)
    - a, b, ρ, m, σ are the SVI parameters
    """
    
    def __init__(self):
        """Initialize the SVI model."""
        self.parameters = None
        self.calibration_result = None
        self._bounds = self._get_default_bounds()
    
    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        """
        Get default parameter bounds for SVI calibration.
        
        Returns:
            List of (min, max) tuples for each parameter [a, b, rho, m, sigma]
        """
        return [
            (0.001, 1.0),    # a: minimum variance level
            (0.001, 2.0),    # b: variance slope
            (-0.999, 0.999), # rho: correlation parameter
            (-2.0, 2.0),     # m: ATM level
            (0.001, 2.0)     # sigma: variance convexity
        ]
    
    def svi_formula(self, log_moneyness: Union[float, np.ndarray], 
                   parameters: List[float]) -> Union[float, np.ndarray]:
        """
        Calculate implied variance using SVI formula.
        
        Args:
            log_moneyness: Log-moneyness values ln(K/F)
            parameters: SVI parameters [a, b, rho, m, sigma]
            
        Returns:
            Implied variance values
        """
        a, b, rho, m, sigma = parameters
        k = np.asarray(log_moneyness)
        
        # SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)² + sigma²))
        k_minus_m = k - m
        sqrt_term = np.sqrt(k_minus_m**2 + sigma**2)
        
        return a + b * (rho * k_minus_m + sqrt_term)
    
    def svi_volatility(self, log_moneyness: Union[float, np.ndarray], 
                      parameters: List[float]) -> Union[float, np.ndarray]:
        """
        Calculate implied volatility using SVI formula.
        
        Args:
            log_moneyness: Log-moneyness values ln(K/F)
            parameters: SVI parameters [a, b, rho, m, sigma]
            
        Returns:
            Implied volatility values
        """
        variance = self.svi_formula(log_moneyness, parameters)
        return np.sqrt(np.maximum(variance, 1e-8))  # Ensure positive variance
    
    def _validate_parameters(self, parameters: List[float]) -> bool:
        """
        Validate SVI parameters to ensure no-arbitrage conditions.
        
        Args:
            parameters: SVI parameters [a, b, rho, m, sigma]
            
        Returns:
            True if parameters are valid, False otherwise
        """
        a, b, rho, m, sigma = parameters
        
        # Basic bounds
        if a <= 0 or b <= 0 or sigma <= 0:
            return False
        if abs(rho) >= 1:
            return False
        
        # No-arbitrage conditions
        # 1. a + b * sigma * (1 + |rho|) >= 0 (ensures positive variance)
        if a + b * sigma * (1 + abs(rho)) < 0:
            return False
        
        # 2. b * (1 + rho) >= 0 and b * (1 - rho) >= 0
        if b * (1 + rho) < 0 or b * (1 - rho) < 0:
            return False
        
        return True
    
    def _objective_function(self, parameters: List[float], log_moneyness: np.ndarray, 
                           target_variances: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Objective function for SVI calibration.
        
        Args:
            parameters: SVI parameters to optimize
            log_moneyness: Log-moneyness values
            target_variances: Target implied variances
            weights: Optional weights for each data point
            
        Returns:
            Sum of squared errors
        """
        if not self._validate_parameters(parameters):
            return 1e10  # Large penalty for invalid parameters
        
        model_variances = self.svi_formula(log_moneyness, parameters)
        errors = model_variances - target_variances
        
        if weights is not None:
            errors = errors * weights
        
        return np.sum(errors**2)
    
    def calibrate(self, option_data: OptionData, method: str = 'least_squares',
                 initial_guess: Optional[List[float]] = None, 
                 weights: Optional[List[float]] = None) -> Dict:
        """
        Calibrate SVI parameters to market data.
        
        Args:
            option_data: Market option data
            method: Optimization method ('least_squares' or 'differential_evolution')
            initial_guess: Initial parameter guess [a, b, rho, m, sigma]
            weights: Optional weights for each data point
            
        Returns:
            Calibration results dictionary
        """
        # Calculate log-moneyness and target variances
        log_moneyness = np.array([np.log(k / option_data.spot) for k in option_data.strikes])
        target_variances = np.array([iv**2 for iv in option_data.implied_volatilities])
        
        weights_array = np.array(weights) if weights is not None else None
        
        if method == 'least_squares':
            result = self._calibrate_least_squares(log_moneyness, target_variances, 
                                                 initial_guess, weights_array)
        elif method == 'differential_evolution':
            result = self._calibrate_differential_evolution(log_moneyness, target_variances, 
                                                          weights_array)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        self.parameters = result['parameters']
        self.calibration_result = result
        
        return result
    
    def _calibrate_least_squares(self, log_moneyness: np.ndarray, target_variances: np.ndarray,
                                initial_guess: Optional[List[float]], 
                                weights: Optional[np.ndarray]) -> Dict:
        """
        Calibrate using least squares optimization.
        
        Args:
            log_moneyness: Log-moneyness values
            target_variances: Target implied variances
            initial_guess: Initial parameter guess
            weights: Optional weights
            
        Returns:
            Calibration results
        """
        if initial_guess is None:
            # Simple initial guess based on data
            initial_guess = [
                np.mean(target_variances),  # a
                0.1,                        # b
                0.0,                        # rho
                0.0,                        # m
                0.1                         # sigma
            ]
        
        result = minimize(
            self._objective_function,
            initial_guess,
            args=(log_moneyness, target_variances, weights),
            bounds=self._bounds,
            method='L-BFGS-B'
        )
        
        return {
            'parameters': result.x.tolist(),
            'success': result.success,
            'objective_value': result.fun,
            'message': result.message,
            'method': 'least_squares'
        }
    
    def _calibrate_differential_evolution(self, log_moneyness: np.ndarray, 
                                        target_variances: np.ndarray,
                                        weights: Optional[np.ndarray]) -> Dict:
        """
        Calibrate using differential evolution optimization.
        
        Args:
            log_moneyness: Log-moneyness values
            target_variances: Target implied variances
            weights: Optional weights
            
        Returns:
            Calibration results
        """
        result = differential_evolution(
            self._objective_function,
            bounds=self._bounds,
            args=(log_moneyness, target_variances, weights),
            seed=42,
            maxiter=1000
        )
        
        return {
            'parameters': result.x.tolist(),
            'success': result.success,
            'objective_value': result.fun,
            'message': result.message,
            'method': 'differential_evolution'
        }
    
    def predict(self, log_moneyness: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Predict implied volatilities for given log-moneyness values.
        
        Args:
            log_moneyness: Log-moneyness values to predict
            
        Returns:
            Predicted implied volatilities
            
        Raises:
            ValueError: If model has not been calibrated
        """
        if self.parameters is None:
            raise ValueError("Model must be calibrated before making predictions")
        
        return self.svi_volatility(log_moneyness, self.parameters)
    
    def get_parameter_names(self) -> List[str]:
        """
        Get parameter names for the SVI model.
        
        Returns:
            List of parameter names
        """
        return ['a', 'b', 'rho', 'm', 'sigma']
