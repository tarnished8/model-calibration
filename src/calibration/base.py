"""
Base calibration classes and interfaces.

This module provides abstract base classes and common functionality
for model calibration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models that can be calibrated.
    """
    
    def __init__(self):
        """Initialize the base model."""
        self.parameters = None
        self.calibration_result = None
    
    @abstractmethod
    def predict(self, inputs: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Make predictions using the calibrated model.
        
        Args:
            inputs: Input values for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """
        Get the names of model parameters.
        
        Returns:
            List of parameter names
        """
        pass
    
    def is_calibrated(self) -> bool:
        """
        Check if the model has been calibrated.
        
        Returns:
            True if model is calibrated, False otherwise
        """
        return self.parameters is not None


class BaseCalibrator(ABC):
    """
    Abstract base class for model calibrators.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize the calibrator.
        
        Args:
            model: Model to be calibrated
        """
        self.model = model
        self.calibration_history = []
    
    @abstractmethod
    def calibrate(self, data: Any, **kwargs) -> Dict:
        """
        Calibrate the model to data.
        
        Args:
            data: Data to calibrate against
            **kwargs: Additional calibration parameters
            
        Returns:
            Calibration results
        """
        pass
    
    def get_calibration_summary(self) -> Dict:
        """
        Get a summary of the calibration results.
        
        Returns:
            Summary dictionary
        """
        if not self.model.is_calibrated():
            return {"status": "not_calibrated"}
        
        return {
            "status": "calibrated",
            "parameters": dict(zip(self.model.get_parameter_names(), self.model.parameters)),
            "calibration_result": self.model.calibration_result
        }


class CalibrationMetrics:
    """
    Utility class for calculating calibration metrics and goodness-of-fit measures.
    """
    
    @staticmethod
    def rmse(predicted: np.ndarray, actual: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            weights: Optional weights for each observation
            
        Returns:
            RMSE value
        """
        errors = predicted - actual
        if weights is not None:
            errors = errors * weights
        return np.sqrt(np.mean(errors**2))
    
    @staticmethod
    def mae(predicted: np.ndarray, actual: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            weights: Optional weights for each observation
            
        Returns:
            MAE value
        """
        errors = np.abs(predicted - actual)
        if weights is not None:
            errors = errors * weights
        return np.mean(errors)
    
    @staticmethod
    def mape(predicted: np.ndarray, actual: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            weights: Optional weights for each observation
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.inf
        
        errors = np.abs((predicted[mask] - actual[mask]) / actual[mask]) * 100
        if weights is not None:
            errors = errors * weights[mask]
        return np.mean(errors)
    
    @staticmethod
    def r_squared(predicted: np.ndarray, actual: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            predicted: Predicted values
            actual: Actual values
            weights: Optional weights for each observation
            
        Returns:
            R-squared value
        """
        if weights is not None:
            actual_mean = np.average(actual, weights=weights)
            ss_tot = np.sum(weights * (actual - actual_mean)**2)
            ss_res = np.sum(weights * (actual - predicted)**2)
        else:
            actual_mean = np.mean(actual)
            ss_tot = np.sum((actual - actual_mean)**2)
            ss_res = np.sum((actual - predicted)**2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_all_metrics(predicted: np.ndarray, actual: np.ndarray, 
                            weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            weights: Optional weights for each observation
            
        Returns:
            Dictionary of all metrics
        """
        return {
            'rmse': CalibrationMetrics.rmse(predicted, actual, weights),
            'mae': CalibrationMetrics.mae(predicted, actual, weights),
            'mape': CalibrationMetrics.mape(predicted, actual, weights),
            'r_squared': CalibrationMetrics.r_squared(predicted, actual, weights)
        }


class CalibrationResult:
    """
    Container class for calibration results with additional analysis.
    """
    
    def __init__(self, parameters: List[float], parameter_names: List[str], 
                 success: bool, objective_value: float, method: str, 
                 predicted_values: Optional[np.ndarray] = None,
                 actual_values: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 additional_info: Optional[Dict] = None):
        """
        Initialize calibration result.
        
        Args:
            parameters: Calibrated parameter values
            parameter_names: Names of parameters
            success: Whether calibration was successful
            objective_value: Final objective function value
            method: Calibration method used
            predicted_values: Model predictions (optional)
            actual_values: Actual target values (optional)
            weights: Weights used in calibration (optional)
            additional_info: Additional information (optional)
        """
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.success = success
        self.objective_value = objective_value
        self.method = method
        self.predicted_values = predicted_values
        self.actual_values = actual_values
        self.weights = weights
        self.additional_info = additional_info or {}
        
        # Calculate metrics if data is available
        self.metrics = None
        if predicted_values is not None and actual_values is not None:
            self.metrics = CalibrationMetrics.calculate_all_metrics(
                predicted_values, actual_values, weights
            )
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """
        Get parameters as a dictionary.
        
        Returns:
            Dictionary mapping parameter names to values
        """
        return dict(zip(self.parameter_names, self.parameters))
    
    def summary(self) -> Dict:
        """
        Get a summary of the calibration result.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'success': self.success,
            'method': self.method,
            'objective_value': self.objective_value,
            'parameters': self.get_parameter_dict()
        }
        
        if self.metrics is not None:
            summary['metrics'] = self.metrics
        
        if self.additional_info:
            summary['additional_info'] = self.additional_info
        
        return summary
