"""
Optimization algorithms for model calibration.

This module provides various optimization algorithms that can be used
for calibrating financial models.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
from typing import Dict, List, Optional, Callable, Tuple, Any
from .base import BaseCalibrator, CalibrationResult


class LeastSquaresOptimizer(BaseCalibrator):
    """
    Least squares optimizer for model calibration.
    
    Uses scipy's minimize function with L-BFGS-B algorithm for bounded optimization.
    """
    
    def __init__(self, model, bounds: Optional[List[Tuple[float, float]]] = None,
                 method: str = 'L-BFGS-B', options: Optional[Dict] = None):
        """
        Initialize the least squares optimizer.
        
        Args:
            model: Model to be calibrated
            bounds: Parameter bounds as list of (min, max) tuples
            method: Optimization method for scipy.minimize
            options: Additional options for the optimizer
        """
        super().__init__(model)
        self.bounds = bounds
        self.method = method
        self.options = options or {}
    
    def calibrate(self, data: Any, objective_function: Callable,
                 initial_guess: List[float], weights: Optional[np.ndarray] = None,
                 **kwargs) -> CalibrationResult:
        """
        Calibrate the model using least squares optimization.
        
        Args:
            data: Data to calibrate against
            objective_function: Function to minimize
            initial_guess: Initial parameter guess
            weights: Optional weights for observations
            **kwargs: Additional arguments passed to objective function
            
        Returns:
            CalibrationResult object
        """
   
        attempt = {
            'method': 'least_squares',
            'initial_guess': initial_guess.copy(),
            'bounds': self.bounds
        }
        
        try:
            result = minimize(
                objective_function,
                initial_guess,
                args=(data, weights, *kwargs.values()) if kwargs else (data, weights),
                bounds=self.bounds,
                method=self.method,
                options=self.options
            )
            
            attempt['success'] = result.success
            attempt['final_parameters'] = result.x.tolist()
            attempt['objective_value'] = result.fun
            attempt['message'] = result.message
            
            # Update model parameters
            self.model.parameters = result.x.tolist()
            
            # Create calibration result
            calibration_result = CalibrationResult(
                parameters=result.x.tolist(),
                parameter_names=self.model.get_parameter_names(),
                success=result.success,
                objective_value=result.fun,
                method='least_squares',
                additional_info={
                    'scipy_result': result,
                    'iterations': result.get('nit', None),
                    'function_evaluations': result.get('nfev', None)
                }
            )
            
            self.model.calibration_result = calibration_result
            
        except Exception as e:
            attempt['success'] = False
            attempt['error'] = str(e)
            
            calibration_result = CalibrationResult(
                parameters=initial_guess,
                parameter_names=self.model.get_parameter_names(),
                success=False,
                objective_value=np.inf,
                method='least_squares',
                additional_info={'error': str(e)}
            )
        
        self.calibration_history.append(attempt)
        return calibration_result


class DifferentialEvolutionOptimizer(BaseCalibrator):
    """
    Differential Evolution optimizer for model calibration.
    
    Uses scipy's differential_evolution for global optimization.
    """
    
    def __init__(self, model, bounds: List[Tuple[float, float]],
                 strategy: str = 'best1bin', maxiter: int = 1000,
                 popsize: int = 15, seed: Optional[int] = None,
                 options: Optional[Dict] = None):
        """
        Initialize the differential evolution optimizer.
        
        Args:
            model: Model to be calibrated
            bounds: Parameter bounds as list of (min, max) tuples
            strategy: DE strategy to use
            maxiter: Maximum number of iterations
            popsize: Population size multiplier
            seed: Random seed for reproducibility
            options: Additional options for the optimizer
        """
        super().__init__(model)
        self.bounds = bounds
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.options = options or {}
    
    def calibrate(self, data: Any, objective_function: Callable,
                 weights: Optional[np.ndarray] = None, **kwargs) -> CalibrationResult:
        """
        Calibrate the model using differential evolution.
        
        Args:
            data: Data to calibrate against
            objective_function: Function to minimize
            weights: Optional weights for observations
            **kwargs: Additional arguments passed to objective function
            
        Returns:
            CalibrationResult object
        """
        # Store calibration attempt
        attempt = {
            'method': 'differential_evolution',
            'bounds': self.bounds,
            'strategy': self.strategy,
            'maxiter': self.maxiter,
            'popsize': self.popsize,
            'seed': self.seed
        }
        
        try:
            result = differential_evolution(
                objective_function,
                bounds=self.bounds,
                args=(data, weights, *kwargs.values()) if kwargs else (data, weights),
                strategy=self.strategy,
                maxiter=self.maxiter,
                popsize=self.popsize,
                seed=self.seed,
                **self.options
            )
            
            attempt['success'] = result.success
            attempt['final_parameters'] = result.x.tolist()
            attempt['objective_value'] = result.fun
            attempt['message'] = result.message
            
            # Update model parameters
            self.model.parameters = result.x.tolist()
            
            # Create calibration result
            calibration_result = CalibrationResult(
                parameters=result.x.tolist(),
                parameter_names=self.model.get_parameter_names(),
                success=result.success,
                objective_value=result.fun,
                method='differential_evolution',
                additional_info={
                    'scipy_result': result,
                    'iterations': result.nit,
                    'function_evaluations': result.nfev
                }
            )
            
            self.model.calibration_result = calibration_result
            
        except Exception as e:
            attempt['success'] = False
            attempt['error'] = str(e)
            
            calibration_result = CalibrationResult(
                parameters=[],
                parameter_names=self.model.get_parameter_names(),
                success=False,
                objective_value=np.inf,
                method='differential_evolution',
                additional_info={'error': str(e)}
            )
        
        self.calibration_history.append(attempt)
        return calibration_result


class NonlinearLeastSquaresOptimizer(BaseCalibrator):
    """
    Nonlinear least squares optimizer using scipy's least_squares.
    
    Particularly useful for problems where the residuals (not the objective)
    are the primary concern.
    """
    
    def __init__(self, model, bounds: Optional[List[Tuple[float, float]]] = None,
                 method: str = 'trf', options: Optional[Dict] = None):
        """
        Initialize the nonlinear least squares optimizer.
        
        Args:
            model: Model to be calibrated
            bounds: Parameter bounds as list of (min, max) tuples
            method: Method for least_squares ('trf', 'dogbox', or 'lm')
            options: Additional options for the optimizer
        """
        super().__init__(model)
        self.bounds = bounds
        self.method = method
        self.options = options or {}
    
    def calibrate(self, data: Any, residual_function: Callable,
                 initial_guess: List[float], weights: Optional[np.ndarray] = None,
                 **kwargs) -> CalibrationResult:
        """
        Calibrate the model using nonlinear least squares.
        
        Args:
            data: Data to calibrate against
            residual_function: Function that returns residuals
            initial_guess: Initial parameter guess
            weights: Optional weights for observations
            **kwargs: Additional arguments passed to residual function
            
        Returns:
            CalibrationResult object
        """
        # Prepare bounds for least_squares format
        if self.bounds is not None:
            bounds_lower = [b[0] for b in self.bounds]
            bounds_upper = [b[1] for b in self.bounds]
            bounds = (bounds_lower, bounds_upper)
        else:
            bounds = (-np.inf, np.inf)
        
        # Store calibration attempt
        attempt = {
            'method': 'nonlinear_least_squares',
            'initial_guess': initial_guess.copy(),
            'bounds': self.bounds
        }
        
        try:
            result = least_squares(
                residual_function,
                initial_guess,
                args=(data, weights, *kwargs.values()) if kwargs else (data, weights),
                bounds=bounds,
                method=self.method,
                **self.options
            )
            
            attempt['success'] = result.success
            attempt['final_parameters'] = result.x.tolist()
            attempt['objective_value'] = 0.5 * np.sum(result.fun**2)  # Convert residuals to objective
            attempt['message'] = result.message
            
            # Update model parameters
            self.model.parameters = result.x.tolist()
            
            # Create calibration result
            calibration_result = CalibrationResult(
                parameters=result.x.tolist(),
                parameter_names=self.model.get_parameter_names(),
                success=result.success,
                objective_value=0.5 * np.sum(result.fun**2),
                method='nonlinear_least_squares',
                additional_info={
                    'scipy_result': result,
                    'function_evaluations': result.nfev,
                    'jacobian_evaluations': result.njev,
                    'optimality': result.optimality,
                    'active_mask': result.active_mask.tolist() if hasattr(result, 'active_mask') else None
                }
            )
            
            self.model.calibration_result = calibration_result
            
        except Exception as e:
            attempt['success'] = False
            attempt['error'] = str(e)
            
            calibration_result = CalibrationResult(
                parameters=initial_guess,
                parameter_names=self.model.get_parameter_names(),
                success=False,
                objective_value=np.inf,
                method='nonlinear_least_squares',
                additional_info={'error': str(e)}
            )
        
        self.calibration_history.append(attempt)
        return calibration_result
