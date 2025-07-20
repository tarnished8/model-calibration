"""
Implied volatility calculation using Newton-Raphson method.

This module provides functionality to calculate implied volatility from market prices
using the Newton-Raphson iterative method.
"""

import numpy as np
from typing import Optional
from .black_scholes import BlackScholes


class ImpliedVolatility:
    """
    Implied volatility calculator using Newton-Raphson method.
    
    This class implements the Newton-Raphson algorithm to find the implied volatility
    that makes the Black-Scholes theoretical price equal to the market price.
    """
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        """
        Initialize the implied volatility calculator.
        
        Args:
            max_iterations: Maximum number of iterations for Newton-Raphson
            tolerance: Convergence tolerance for the algorithm
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.bs_model = BlackScholes()
    
    def calculate(self, spot: float, strike: float, time_to_maturity: float,
                 risk_free_rate: float, market_price: float, 
                 option_type: str = 'call', initial_guess: Optional[float] = None) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            market_price: Market price of the option
            option_type: Type of option ('call' or 'put')
            initial_guess: Initial guess for volatility (if None, will be estimated)
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If convergence is not achieved or inputs are invalid
        """
        if market_price <= 0:
            raise ValueError("Market price must be positive")
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if spot <= 0:
            raise ValueError("Spot price must be positive")
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        
        # Set initial guess if not provided
        if initial_guess is None:
            initial_guess = self._get_initial_guess(spot, strike, time_to_maturity, market_price)
        
        volatility = initial_guess
        
        for iteration in range(self.max_iterations):
            try:
                # Calculate Black-Scholes price and vega
                bs_result = self.bs_model.price(
                    spot=spot,
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    option_type=option_type
                )
                
                price_diff = bs_result['price'] - market_price
                vega = bs_result['vega']
                
                # Check for convergence
                if abs(price_diff) < self.tolerance:
                    return volatility
                
                # Check if vega is too small (avoid division by zero)
                if abs(vega) < 1e-10:
                    raise ValueError("Vega is too small, cannot continue Newton-Raphson")
                
                # Newton-Raphson update
                volatility_new = volatility - price_diff / vega
                
                # Ensure volatility stays positive
                if volatility_new <= 0:
                    volatility_new = volatility / 2
                
                # Check for convergence in volatility
                if abs(volatility_new - volatility) < self.tolerance:
                    return volatility_new
                
                volatility = volatility_new
                
            except (ValueError, ZeroDivisionError) as e:
                # If we encounter numerical issues, try a different starting point
                if iteration == 0:
                    volatility = 0.2  # Standard fallback
                    continue
                else:
                    raise ValueError(f"Failed to converge: {str(e)}")
        
        raise ValueError(f"Failed to converge after {self.max_iterations} iterations")
    
    def _get_initial_guess(self, spot: float, strike: float, 
                          time_to_maturity: float, market_price: float) -> float:
        """
        Get initial guess for implied volatility.
        
        Uses Brenner-Subrahmanyam approximation as a starting point.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            market_price: Market price of the option
            
        Returns:
            Initial guess for volatility
        """
        # Brenner-Subrahmanyam approximation
        guess = np.sqrt(2 * np.pi / time_to_maturity) * market_price / spot
        
        # Ensure reasonable bounds
        guess = max(0.01, min(5.0, guess))
        
        return guess
    
    def calculate_multiple(self, spot: float, strikes: list, time_to_maturity: float,
                          risk_free_rate: float, market_prices: list,
                          option_type: str = 'call') -> list:
        """
        Calculate implied volatilities for multiple strikes.
        
        Args:
            spot: Current price of the underlying asset
            strikes: List of strike prices
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            market_prices: List of market prices corresponding to strikes
            option_type: Type of option ('call' or 'put')
            
        Returns:
            List of implied volatilities
            
        Raises:
            ValueError: If strikes and prices lists have different lengths
        """
        if len(strikes) != len(market_prices):
            raise ValueError("Strikes and market prices must have the same length")
        
        implied_vols = []
        
        for strike, market_price in zip(strikes, market_prices):
            try:
                iv = self.calculate(
                    spot=spot,
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    market_price=market_price,
                    option_type=option_type
                )
                implied_vols.append(iv)
            except ValueError as e:
                print(f"Warning: Failed to calculate IV for strike {strike}: {str(e)}")
                implied_vols.append(np.nan)
        
        return implied_vols
