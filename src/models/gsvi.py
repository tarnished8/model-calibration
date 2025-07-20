"""
gSVI (generalized SVI) model implementation.

This module provides the generalized SVI model for volatility surface modeling
with enhanced parameterization and calibration capabilities.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Union
from .svi import SVI
from ..data.option_data import OptionData


class gSVI(SVI):
    """
    Generalized SVI model for volatility surface modeling.
    
    The gSVI model extends the SVI model with additional parameterizations
    and improved calibration methods for better fitting of volatility surfaces.
    """
    
    def __init__(self, parameterization: str = 'raw'):
        """
        Initialize the gSVI model.
        
        Args:
            parameterization: Type of parameterization ('raw', 'natural', or 'jump_wings')
        """
        super().__init__()
        self.parameterization = parameterization
        self._set_bounds_for_parameterization()
    
    def _set_bounds_for_parameterization(self):
        """Set parameter bounds based on the chosen parameterization."""
        if self.parameterization == 'raw':
            self._bounds = self._get_default_bounds()
        elif self.parameterization == 'natural':
            self._bounds = [
                (0.001, 1.0),    # delta (ATM skew)
                (0.001, 2.0),    # mu (ATM curvature)
                (-0.999, 0.999), # rho (correlation)
                (0.001, 2.0),    # omega (ATM variance)
                (0.001, 2.0)     # zeta (variance of variance)
            ]
        elif self.parameterization == 'jump_wings':
            self._bounds = [
                (0.001, 1.0),    # v_t (ATM variance)
                (0.001, 2.0),    # psi (left wing)
                (0.001, 2.0),    # p (right wing)
                (-2.0, 2.0),     # c (shift)
                (0.001, 2.0)     # v_min (minimum variance)
            ]
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def _convert_to_raw_parameters(self, parameters: List[float]) -> List[float]:
        """
        Convert parameters from chosen parameterization to raw SVI parameters.
        
        Args:
            parameters: Parameters in the chosen parameterization
            
        Returns:
            Parameters in raw SVI format [a, b, rho, m, sigma]
        """
        if self.parameterization == 'raw':
            return parameters
        elif self.parameterization == 'natural':
            return self._natural_to_raw(parameters)
        elif self.parameterization == 'jump_wings':
            return self._jump_wings_to_raw(parameters)
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def _natural_to_raw(self, natural_params: List[float]) -> List[float]:
        """
        Convert natural parameterization to raw SVI parameters.
        
        Natural parameterization: [delta, mu, rho, omega, zeta]
        Raw parameterization: [a, b, rho, m, sigma]
        
        Args:
            natural_params: Parameters in natural form
            
        Returns:
            Parameters in raw SVI form
        """
        delta, mu, rho, omega, zeta = natural_params
        
        # Convert to raw parameters
        a = omega / 2
        b = mu * omega / (2 * zeta)
        m = delta / mu
        sigma = zeta / mu
        
        return [a, b, rho, m, sigma]
    
    def _jump_wings_to_raw(self, jw_params: List[float]) -> List[float]:
        """
        Convert jump-wings parameterization to raw SVI parameters.
        
        Jump-wings parameterization: [v_t, psi, p, c, v_min]
        Raw parameterization: [a, b, rho, m, sigma]
        
        Args:
            jw_params: Parameters in jump-wings form
            
        Returns:
            Parameters in raw SVI form
        """
        v_t, psi, p, c, v_min = jw_params
        
        # Convert to raw parameters (simplified conversion)
        a = v_min
        b = (v_t - v_min) / 2
        rho = (psi - p) / (psi + p)
        m = c
        sigma = np.sqrt((psi + p) / 2)
        
        return [a, b, rho, m, sigma]
    
    def gsvi_formula(self, log_moneyness: Union[float, np.ndarray], 
                    parameters: List[float]) -> Union[float, np.ndarray]:
        """
        Calculate implied variance using gSVI formula.
        
        Args:
            log_moneyness: Log-moneyness values ln(K/F)
            parameters: gSVI parameters in the chosen parameterization
            
        Returns:
            Implied variance values
        """
        raw_params = self._convert_to_raw_parameters(parameters)
        return self.svi_formula(log_moneyness, raw_params)
    
    def gsvi_volatility(self, log_moneyness: Union[float, np.ndarray], 
                       parameters: List[float]) -> Union[float, np.ndarray]:
        """
        Calculate implied volatility using gSVI formula.
        
        Args:
            log_moneyness: Log-moneyness values ln(K/F)
            parameters: gSVI parameters in the chosen parameterization
            
        Returns:
            Implied volatility values
        """
        variance = self.gsvi_formula(log_moneyness, parameters)
        return np.sqrt(np.maximum(variance, 1e-8))
    
    def _validate_gsvi_parameters(self, parameters: List[float]) -> bool:
        """
        Validate gSVI parameters based on the chosen parameterization.
        
        Args:
            parameters: gSVI parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            raw_params = self._convert_to_raw_parameters(parameters)
            return self._validate_parameters(raw_params)
        except (ValueError, ZeroDivisionError):
            return False
    
    def _gsvi_objective_function(self, parameters: List[float], log_moneyness: np.ndarray, 
                                target_variances: np.ndarray, 
                                weights: Optional[np.ndarray] = None) -> float:
        """
        Objective function for gSVI calibration.
        
        Args:
            parameters: gSVI parameters to optimize
            log_moneyness: Log-moneyness values
            target_variances: Target implied variances
            weights: Optional weights for each data point
            
        Returns:
            Sum of squared errors
        """
        if not self._validate_gsvi_parameters(parameters):
            return 1e10
        
        model_variances = self.gsvi_formula(log_moneyness, parameters)
        errors = model_variances - target_variances
        
        if weights is not None:
            errors = errors * weights
        
        return np.sum(errors**2)
    
    def calibrate(self, option_data: OptionData, method: str = 'differential_evolution',
                 initial_guess: Optional[List[float]] = None, 
                 weights: Optional[List[float]] = None) -> Dict:
        """
        Calibrate gSVI parameters to market data.
        
        Args:
            option_data: Market option data
            method: Optimization method ('least_squares' or 'differential_evolution')
            initial_guess: Initial parameter guess
            weights: Optional weights for each data point
            
        Returns:
            Calibration results dictionary
        """
        log_moneyness = np.array([np.log(k / option_data.spot) for k in option_data.strikes])
        target_variances = np.array([iv**2 for iv in option_data.implied_volatilities])
        
        weights_array = np.array(weights) if weights is not None else None
        
        if method == 'least_squares':
            result = self._calibrate_gsvi_least_squares(log_moneyness, target_variances, 
                                                       initial_guess, weights_array)
        elif method == 'differential_evolution':
            result = self._calibrate_gsvi_differential_evolution(log_moneyness, target_variances, 
                                                               weights_array)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        self.parameters = result['parameters']
        self.calibration_result = result
        
        return result
    
    def _calibrate_gsvi_least_squares(self, log_moneyness: np.ndarray, target_variances: np.ndarray,
                                     initial_guess: Optional[List[float]], 
                                     weights: Optional[np.ndarray]) -> Dict:
        """Calibrate using least squares optimization for gSVI."""
        if initial_guess is None:
            initial_guess = self._get_default_initial_guess(target_variances)
        
        result = minimize(
            self._gsvi_objective_function,
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
            'method': 'least_squares',
            'parameterization': self.parameterization
        }
    
    def _calibrate_gsvi_differential_evolution(self, log_moneyness: np.ndarray, 
                                             target_variances: np.ndarray,
                                             weights: Optional[np.ndarray]) -> Dict:
        """Calibrate using differential evolution optimization for gSVI."""
        result = differential_evolution(
            self._gsvi_objective_function,
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
            'method': 'differential_evolution',
            'parameterization': self.parameterization
        }
    
    def _get_default_initial_guess(self, target_variances: np.ndarray) -> List[float]:
        """Get default initial guess based on parameterization."""
        if self.parameterization == 'raw':
            return [np.mean(target_variances), 0.1, 0.0, 0.0, 0.1]
        elif self.parameterization == 'natural':
            return [0.1, 0.5, 0.0, np.mean(target_variances), 0.1]
        elif self.parameterization == 'jump_wings':
            return [np.mean(target_variances), 0.1, 0.1, 0.0, np.min(target_variances)]
        
    def predict(self, log_moneyness: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Predict implied volatilities using gSVI model."""
        if self.parameters is None:
            raise ValueError("Model must be calibrated before making predictions")
        
        return self.gsvi_volatility(log_moneyness, self.parameters)
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names for the chosen parameterization."""
        if self.parameterization == 'raw':
            return ['a', 'b', 'rho', 'm', 'sigma']
        elif self.parameterization == 'natural':
            return ['delta', 'mu', 'rho', 'omega', 'zeta']
        elif self.parameterization == 'jump_wings':
            return ['v_t', 'psi', 'p', 'c', 'v_min']
