"""
Option data structures and utilities.

This module provides classes for handling option market data,
including validation and preprocessing functionality.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from ..models.implied_volatility import ImpliedVolatility


@dataclass
class OptionData:
    """
    Container for option market data.
    
    This class holds option market data including strikes, prices, and market parameters,
    with automatic calculation of implied volatilities and log-moneyness.
    """
    
    strikes: List[float]
    prices: List[float]
    spot: float
    risk_free_rate: float = 0.0
    time_to_maturity: float = 1.0
    option_type: str = 'call'
    implied_volatilities: Optional[List[float]] = field(default=None, init=False)
    log_moneyness: Optional[List[float]] = field(default=None, init=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._validate_data()
        self._calculate_derived_quantities()
    
    def _validate_data(self):
        """Validate the input data."""
        if len(self.strikes) != len(self.prices):
            raise ValueError("Strikes and prices must have the same length")
        
        if len(self.strikes) == 0:
            raise ValueError("Must provide at least one strike-price pair")
        
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        
        if any(k <= 0 for k in self.strikes):
            raise ValueError("All strikes must be positive")
        
        if any(p <= 0 for p in self.prices):
            raise ValueError("All prices must be positive")
        
        if self.option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def _calculate_derived_quantities(self):
        """Calculate implied volatilities and log-moneyness."""
        # Calculate log-moneyness
        self.log_moneyness = [np.log(k / self.spot) for k in self.strikes]
        
        # Calculate implied volatilities
        iv_calculator = ImpliedVolatility()
        self.implied_volatilities = []
        
        for strike, price in zip(self.strikes, self.prices):
            try:
                iv = iv_calculator.calculate(
                    spot=self.spot,
                    strike=strike,
                    time_to_maturity=self.time_to_maturity,
                    risk_free_rate=self.risk_free_rate,
                    market_price=price,
                    option_type=self.option_type
                )
                self.implied_volatilities.append(iv)
            except ValueError as e:
                print(f"Warning: Could not calculate IV for strike {strike}: {e}")
                self.implied_volatilities.append(np.nan)
    
    def get_valid_data(self) -> 'OptionData':
        """
        Get a copy of the data with only valid (non-NaN) implied volatilities.
        
        Returns:
            New OptionData instance with valid data only
        """
        valid_indices = [i for i, iv in enumerate(self.implied_volatilities) 
                        if not np.isnan(iv)]
        
        if not valid_indices:
            raise ValueError("No valid implied volatilities found")
        
        return OptionData(
            strikes=[self.strikes[i] for i in valid_indices],
            prices=[self.prices[i] for i in valid_indices],
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
    
    def filter_by_moneyness(self, min_moneyness: float = -0.5, 
                           max_moneyness: float = 0.5) -> 'OptionData':
        """
        Filter data by log-moneyness range.
        
        Args:
            min_moneyness: Minimum log-moneyness to include
            max_moneyness: Maximum log-moneyness to include
            
        Returns:
            New OptionData instance with filtered data
        """
        valid_indices = [i for i, lm in enumerate(self.log_moneyness)
                        if min_moneyness <= lm <= max_moneyness and 
                        not np.isnan(self.implied_volatilities[i])]
        
        if not valid_indices:
            raise ValueError("No data points in the specified moneyness range")
        
        return OptionData(
            strikes=[self.strikes[i] for i in valid_indices],
            prices=[self.prices[i] for i in valid_indices],
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
    
    def add_weights(self, weight_type: str = 'equal') -> List[float]:
        """
        Generate weights for calibration.
        
        Args:
            weight_type: Type of weighting ('equal', 'vega', 'price')
            
        Returns:
            List of weights
        """
        n = len(self.strikes)
        
        if weight_type == 'equal':
            return [1.0] * n
        elif weight_type == 'vega':
            # Weight by vega (higher vega = higher weight)
            from ..models.black_scholes import BlackScholes
            bs = BlackScholes()
            weights = []
            
            for strike, iv in zip(self.strikes, self.implied_volatilities):
                if np.isnan(iv):
                    weights.append(0.0)
                else:
                    result = bs.price(self.spot, strike, self.time_to_maturity,
                                    self.risk_free_rate, iv, self.option_type)
                    weights.append(result['vega'])
            
            # Normalize weights
            total_weight = sum(weights)
            return [w / total_weight * n for w in weights] if total_weight > 0 else [1.0] * n
            
        elif weight_type == 'price':
            # Weight by price (higher price = higher weight)
            total_price = sum(self.prices)
            return [p / total_price * n for p in self.prices] if total_price > 0 else [1.0] * n
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the option data.
        
        Returns:
            Dictionary with data summary
        """
        valid_ivs = [iv for iv in self.implied_volatilities if not np.isnan(iv)]
        
        return {
            'num_options': len(self.strikes),
            'num_valid_ivs': len(valid_ivs),
            'spot': self.spot,
            'risk_free_rate': self.risk_free_rate,
            'time_to_maturity': self.time_to_maturity,
            'option_type': self.option_type,
            'strike_range': (min(self.strikes), max(self.strikes)),
            'price_range': (min(self.prices), max(self.prices)),
            'moneyness_range': (min(self.log_moneyness), max(self.log_moneyness)),
            'iv_range': (min(valid_ivs), max(valid_ivs)) if valid_ivs else (np.nan, np.nan),
            'iv_mean': np.mean(valid_ivs) if valid_ivs else np.nan,
            'iv_std': np.std(valid_ivs) if valid_ivs else np.nan
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.
        
        Returns:
            Dictionary representation of the data
        """
        return {
            'strikes': self.strikes,
            'prices': self.prices,
            'spot': self.spot,
            'risk_free_rate': self.risk_free_rate,
            'time_to_maturity': self.time_to_maturity,
            'option_type': self.option_type,
            'implied_volatilities': self.implied_volatilities,
            'log_moneyness': self.log_moneyness
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionData':
        """
        Create OptionData from dictionary.
        
        Args:
            data: Dictionary with option data
            
        Returns:
            OptionData instance
        """
        return cls(
            strikes=data['strikes'],
            prices=data['prices'],
            spot=data['spot'],
            risk_free_rate=data.get('risk_free_rate', 0.0),
            time_to_maturity=data.get('time_to_maturity', 1.0),
            option_type=data.get('option_type', 'call')
        )


def create_sample_data() -> OptionData:
    """
    Create sample option data for testing and examples.
    
    Returns:
        Sample OptionData instance
    """
    return OptionData(
        strikes=[95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        prices=[10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46],
        spot=100,
        risk_free_rate=0.002,
        time_to_maturity=1.0,
        option_type='call'
    )
