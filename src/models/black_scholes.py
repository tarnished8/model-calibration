"""
Black-Scholes option pricing model implementation.

This module provides a comprehensive implementation of the Black-Scholes model
for European option pricing, including calculation of Greeks.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Union


class BlackScholes:
    """
    Black-Scholes option pricing model.
    
    This class implements the Black-Scholes formula for European call options
    and provides methods to calculate various Greeks (sensitivities).
    """
    
    def __init__(self):
        """Initialize the Black-Scholes model."""
        pass
    
    def _calculate_d1_d2(self, spot: float, strike: float, time_to_maturity: float, 
                        risk_free_rate: float, volatility: float) -> tuple:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility of the underlying asset
            
        Returns:
            Tuple of (d1, d2) values
        """
        d1 = (np.log(spot / strike) + (risk_free_rate + volatility**2 / 2) * time_to_maturity) / (
            volatility * np.sqrt(time_to_maturity)
        )
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        return d1, d2
    
    def price(self, spot: float, strike: float, time_to_maturity: float,
              risk_free_rate: float, volatility: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option price and Greeks using Black-Scholes formula.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility of the underlying asset
            option_type: Type of option ('call' or 'put')
            
        Returns:
            Dictionary containing price and Greeks
        """
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        if spot <= 0:
            raise ValueError("Spot price must be positive")
        if strike <= 0:
            raise ValueError("Strike price must be positive")
            
        d1, d2 = self._calculate_d1_d2(spot, strike, time_to_maturity, risk_free_rate, volatility)
        
        # Calculate price
        if option_type.lower() == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Calculate Greeks
        greeks = self._calculate_greeks(spot, strike, time_to_maturity, risk_free_rate, 
                                      volatility, d1, d2, option_type)
        
        result = {'price': price}
        result.update(greeks)
        
        return result
    
    def _calculate_greeks(self, spot: float, strike: float, time_to_maturity: float,
                         risk_free_rate: float, volatility: float, d1: float, d2: float,
                         option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility of the underlying asset
            d1: d1 parameter from Black-Scholes formula
            d2: d2 parameter from Black-Scholes formula
            option_type: Type of option ('call' or 'put')
            
        Returns:
            Dictionary containing Greeks
        """
        sqrt_t = np.sqrt(time_to_maturity)
        exp_rt = np.exp(-risk_free_rate * time_to_maturity)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (volatility * spot * sqrt_t)
        
        # Vega (same for calls and puts)
        vega = spot * sqrt_t * norm.pdf(d1)
        
        # Theta
        if option_type.lower() == 'call':
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) - 
                    risk_free_rate * strike * exp_rt * norm.cdf(d2))
        else:  # put
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) + 
                    risk_free_rate * strike * exp_rt * norm.cdf(-d2))
        
        # Rho
        if option_type.lower() == 'call':
            rho = strike * time_to_maturity * exp_rt * norm.cdf(d2)
        else:  # put
            rho = -strike * time_to_maturity * exp_rt * norm.cdf(-d2)
        
        # Dual Delta (derivative with respect to strike)
        if option_type.lower() == 'call':
            dual_delta = -exp_rt * norm.cdf(d2)
        else:  # put
            dual_delta = exp_rt * norm.cdf(-d2)
        
        # Dual Gamma
        dual_gamma = -exp_rt * norm.pdf(d2) / (strike * volatility * sqrt_t)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'dual_delta': dual_delta,
            'dual_gamma': dual_gamma
        }
    
    def implied_volatility_initial_guess(self, spot: float, strike: float, 
                                       time_to_maturity: float, market_price: float) -> float:
        """
        Provide an initial guess for implied volatility using Brenner-Subrahmanyam approximation.
        
        Args:
            spot: Current price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            market_price: Market price of the option
            
        Returns:
            Initial guess for implied volatility
        """
        return np.sqrt(2 * np.pi / time_to_maturity) * market_price / spot
