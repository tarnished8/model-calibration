"""
Parameter constraints for model calibration.

This module provides utilities for defining and enforcing parameter constraints
during model calibration, particularly for financial models with no-arbitrage conditions.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod


class ParameterConstraint(ABC):
    """
    Abstract base class for parameter constraints.
    """
    
    @abstractmethod
    def is_satisfied(self, parameters: List[float]) -> bool:
        """
        Check if the constraint is satisfied.
        
        Args:
            parameters: Parameter values to check
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        pass
    
    @abstractmethod
    def penalty(self, parameters: List[float]) -> float:
        """
        Calculate penalty for constraint violation.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Penalty value (0 if constraint is satisfied)
        """
        pass


class BoundConstraint(ParameterConstraint):
    """
    Simple bound constraint for individual parameters.
    """
    
    def __init__(self, parameter_index: int, lower_bound: float, upper_bound: float,
                 penalty_factor: float = 1e6):
        """
        Initialize bound constraint.
        
        Args:
            parameter_index: Index of the parameter to constrain
            lower_bound: Lower bound for the parameter
            upper_bound: Upper bound for the parameter
            penalty_factor: Penalty factor for violations
        """
        self.parameter_index = parameter_index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.penalty_factor = penalty_factor
    
    def is_satisfied(self, parameters: List[float]) -> bool:
        """Check if bound constraint is satisfied."""
        if self.parameter_index >= len(parameters):
            return False
        
        param_value = parameters[self.parameter_index]
        return self.lower_bound <= param_value <= self.upper_bound
    
    def penalty(self, parameters: List[float]) -> float:
        """Calculate penalty for bound violation."""
        if self.parameter_index >= len(parameters):
            return self.penalty_factor
        
        param_value = parameters[self.parameter_index]
        
        if param_value < self.lower_bound:
            return self.penalty_factor * (self.lower_bound - param_value)**2
        elif param_value > self.upper_bound:
            return self.penalty_factor * (param_value - self.upper_bound)**2
        else:
            return 0.0


class LinearConstraint(ParameterConstraint):
    """
    Linear constraint of the form: a^T * x + b >= 0
    """
    
    def __init__(self, coefficients: List[float], constant: float = 0.0,
                 penalty_factor: float = 1e6):
        """
        Initialize linear constraint.
        
        Args:
            coefficients: Coefficients for the linear constraint
            constant: Constant term in the constraint
            penalty_factor: Penalty factor for violations
        """
        self.coefficients = np.array(coefficients)
        self.constant = constant
        self.penalty_factor = penalty_factor
    
    def is_satisfied(self, parameters: List[float]) -> bool:
        """Check if linear constraint is satisfied."""
        if len(parameters) != len(self.coefficients):
            return False
        
        return np.dot(self.coefficients, parameters) + self.constant >= 0
    
    def penalty(self, parameters: List[float]) -> float:
        """Calculate penalty for linear constraint violation."""
        if len(parameters) != len(self.coefficients):
            return self.penalty_factor
        
        constraint_value = np.dot(self.coefficients, parameters) + self.constant
        
        if constraint_value < 0:
            return self.penalty_factor * constraint_value**2
        else:
            return 0.0


class CustomConstraint(ParameterConstraint):
    """
    Custom constraint defined by a user function.
    """
    
    def __init__(self, constraint_function: Callable[[List[float]], bool],
                 penalty_function: Optional[Callable[[List[float]], float]] = None,
                 penalty_factor: float = 1e6):
        """
        Initialize custom constraint.
        
        Args:
            constraint_function: Function that returns True if constraint is satisfied
            penalty_function: Optional function to calculate penalty (if None, uses default)
            penalty_factor: Default penalty factor if penalty_function is None
        """
        self.constraint_function = constraint_function
        self.penalty_function = penalty_function
        self.penalty_factor = penalty_factor
    
    def is_satisfied(self, parameters: List[float]) -> bool:
        """Check if custom constraint is satisfied."""
        try:
            return self.constraint_function(parameters)
        except Exception:
            return False
    
    def penalty(self, parameters: List[float]) -> float:
        """Calculate penalty for custom constraint violation."""
        if self.penalty_function is not None:
            try:
                return self.penalty_function(parameters)
            except Exception:
                return self.penalty_factor
        else:
            return 0.0 if self.is_satisfied(parameters) else self.penalty_factor


class ParameterConstraints:
    """
    Container for multiple parameter constraints with utilities for enforcement.
    """
    
    def __init__(self):
        """Initialize parameter constraints container."""
        self.constraints = []
    
    def add_constraint(self, constraint: ParameterConstraint):
        """
        Add a constraint to the container.
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
    
    def add_bound_constraint(self, parameter_index: int, lower_bound: float, 
                           upper_bound: float, penalty_factor: float = 1e6):
        """
        Add a bound constraint.
        
        Args:
            parameter_index: Index of parameter to constrain
            lower_bound: Lower bound
            upper_bound: Upper bound
            penalty_factor: Penalty factor for violations
        """
        constraint = BoundConstraint(parameter_index, lower_bound, upper_bound, penalty_factor)
        self.add_constraint(constraint)
    
    def add_linear_constraint(self, coefficients: List[float], constant: float = 0.0,
                            penalty_factor: float = 1e6):
        """
        Add a linear constraint.
        
        Args:
            coefficients: Coefficients for linear constraint
            constant: Constant term
            penalty_factor: Penalty factor for violations
        """
        constraint = LinearConstraint(coefficients, constant, penalty_factor)
        self.add_constraint(constraint)
    
    def add_custom_constraint(self, constraint_function: Callable[[List[float]], bool],
                            penalty_function: Optional[Callable[[List[float]], float]] = None,
                            penalty_factor: float = 1e6):
        """
        Add a custom constraint.
        
        Args:
            constraint_function: Function to check constraint satisfaction
            penalty_function: Optional penalty function
            penalty_factor: Default penalty factor
        """
        constraint = CustomConstraint(constraint_function, penalty_function, penalty_factor)
        self.add_constraint(constraint)
    
    def are_satisfied(self, parameters: List[float]) -> bool:
        """
        Check if all constraints are satisfied.
        
        Args:
            parameters: Parameter values to check
            
        Returns:
            True if all constraints are satisfied
        """
        return all(constraint.is_satisfied(parameters) for constraint in self.constraints)
    
    def total_penalty(self, parameters: List[float]) -> float:
        """
        Calculate total penalty for all constraint violations.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Total penalty value
        """
        return sum(constraint.penalty(parameters) for constraint in self.constraints)
    
    def get_violation_summary(self, parameters: List[float]) -> Dict[str, Any]:
        """
        Get a summary of constraint violations.
        
        Args:
            parameters: Parameter values to check
            
        Returns:
            Dictionary with violation information
        """
        violations = []
        total_penalty = 0.0
        
        for i, constraint in enumerate(self.constraints):
            is_satisfied = constraint.is_satisfied(parameters)
            penalty = constraint.penalty(parameters)
            
            violations.append({
                'constraint_index': i,
                'constraint_type': type(constraint).__name__,
                'is_satisfied': is_satisfied,
                'penalty': penalty
            })
            
            total_penalty += penalty
        
        return {
            'all_satisfied': self.are_satisfied(parameters),
            'total_penalty': total_penalty,
            'violations': violations
        }


class SVIConstraints(ParameterConstraints):
    """
    Specialized constraints for SVI model parameters to ensure no-arbitrage conditions.
    """
    
    def __init__(self):
        """Initialize SVI-specific constraints."""
        super().__init__()
        self._setup_svi_constraints()
    
    def _setup_svi_constraints(self):
        """Set up standard SVI no-arbitrage constraints."""
        # Basic positivity constraints
        self.add_bound_constraint(0, 0.001, 1.0)    # a > 0
        self.add_bound_constraint(1, 0.001, 2.0)    # b > 0
        self.add_bound_constraint(2, -0.999, 0.999) # |rho| < 1
        self.add_bound_constraint(3, -2.0, 2.0)     # m bounds
        self.add_bound_constraint(4, 0.001, 2.0)    # sigma > 0
        
        # No-arbitrage constraint: a + b * sigma * (1 + |rho|) >= 0
        def no_arbitrage_1(params):
            a, b, rho, m, sigma = params
            return a + b * sigma * (1 + abs(rho)) >= 0
        
        self.add_custom_constraint(no_arbitrage_1)
        
        # No-arbitrage constraint: b * (1 + rho) >= 0
        def no_arbitrage_2(params):
            a, b, rho, m, sigma = params
            return b * (1 + rho) >= 0
        
        self.add_custom_constraint(no_arbitrage_2)
        
        # No-arbitrage constraint: b * (1 - rho) >= 0
        def no_arbitrage_3(params):
            a, b, rho, m, sigma = params
            return b * (1 - rho) >= 0
        
        self.add_custom_constraint(no_arbitrage_3)
