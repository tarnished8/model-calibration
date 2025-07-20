"""
Market data handling and management.

This module provides classes for managing multiple option datasets,
market data validation, and data preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime, date
from .option_data import OptionData


class MarketData:
    """
    Container for managing multiple option datasets and market information.
    
    This class provides functionality to store, validate, and manipulate
    market data for multiple expiries, strikes, and underlying assets.
    """
    
    def __init__(self, underlying_symbol: str = ""):
        """
        Initialize market data container.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
        """
        self.underlying_symbol = underlying_symbol
        self.option_chains: Dict[str, OptionData] = {}
        self.market_info: Dict[str, Any] = {}
        self.last_updated: Optional[datetime] = None
    
    def add_option_chain(self, expiry: str, option_data: OptionData):
        """
        Add an option chain for a specific expiry.
        
        Args:
            expiry: Expiry identifier (e.g., '2024-01-19')
            option_data: OptionData instance for this expiry
        """
        self.option_chains[expiry] = option_data
        self.last_updated = datetime.now()
    
    def get_option_chain(self, expiry: str) -> Optional[OptionData]:
        """
        Get option chain for a specific expiry.
        
        Args:
            expiry: Expiry identifier
            
        Returns:
            OptionData instance or None if not found
        """
        return self.option_chains.get(expiry)
    
    def get_all_expiries(self) -> List[str]:
        """
        Get all available expiry dates.
        
        Returns:
            List of expiry identifiers
        """
        return list(self.option_chains.keys())
    
    def remove_option_chain(self, expiry: str) -> bool:
        """
        Remove option chain for a specific expiry.
        
        Args:
            expiry: Expiry identifier
            
        Returns:
            True if removed, False if not found
        """
        if expiry in self.option_chains:
            del self.option_chains[expiry]
            self.last_updated = datetime.now()
            return True
        return False
    
    def filter_by_moneyness(self, min_moneyness: float = -0.5, 
                           max_moneyness: float = 0.5) -> 'MarketData':
        """
        Filter all option chains by moneyness range.
        
        Args:
            min_moneyness: Minimum log-moneyness
            max_moneyness: Maximum log-moneyness
            
        Returns:
            New MarketData instance with filtered data
        """
        filtered_data = MarketData(self.underlying_symbol)
        filtered_data.market_info = self.market_info.copy()
        
        for expiry, option_data in self.option_chains.items():
            try:
                filtered_chain = option_data.filter_by_moneyness(min_moneyness, max_moneyness)
                filtered_data.add_option_chain(expiry, filtered_chain)
            except ValueError:
                # Skip expiries with no data in the range
                continue
        
        return filtered_data
    
    def get_atm_volatilities(self) -> Dict[str, float]:
        """
        Get at-the-money volatilities for all expiries.
        
        Returns:
            Dictionary mapping expiry to ATM volatility
        """
        atm_vols = {}
        
        for expiry, option_data in self.option_chains.items():
            # Find the strike closest to ATM
            atm_index = np.argmin([abs(lm) for lm in option_data.log_moneyness])
            
            if not np.isnan(option_data.implied_volatilities[atm_index]):
                atm_vols[expiry] = option_data.implied_volatilities[atm_index]
        
        return atm_vols
    
    def get_volatility_surface(self) -> pd.DataFrame:
        """
        Create a volatility surface DataFrame.
        
        Returns:
            DataFrame with strikes as index, expiries as columns, and IVs as values
        """
        if not self.option_chains:
            return pd.DataFrame()
        
        # Collect all unique strikes
        all_strikes = set()
        for option_data in self.option_chains.values():
            all_strikes.update(option_data.strikes)
        
        all_strikes = sorted(list(all_strikes))
        
        # Create DataFrame
        surface_data = {}
        
        for expiry, option_data in self.option_chains.items():
            strike_to_iv = dict(zip(option_data.strikes, option_data.implied_volatilities))
            surface_data[expiry] = [strike_to_iv.get(strike, np.nan) for strike in all_strikes]
        
        return pd.DataFrame(surface_data, index=all_strikes)
    
    def validate_data(self) -> Dict[str, List[str]]:
        """
        Validate all option data and return issues found.
        
        Returns:
            Dictionary with validation issues by expiry
        """
        issues = {}
        
        for expiry, option_data in self.option_chains.items():
            expiry_issues = []
            
            # Check for NaN implied volatilities
            nan_count = sum(1 for iv in option_data.implied_volatilities if np.isnan(iv))
            if nan_count > 0:
                expiry_issues.append(f"{nan_count} options with invalid implied volatilities")
            
            # Check for arbitrage violations (call prices should be decreasing in strike)
            if option_data.option_type.lower() == 'call':
                for i in range(len(option_data.prices) - 1):
                    if option_data.prices[i] < option_data.prices[i + 1]:
                        expiry_issues.append(f"Call price arbitrage: price increases from strike "
                                           f"{option_data.strikes[i]} to {option_data.strikes[i + 1]}")
            
            # Check for extreme implied volatilities
            valid_ivs = [iv for iv in option_data.implied_volatilities if not np.isnan(iv)]
            if valid_ivs:
                min_iv, max_iv = min(valid_ivs), max(valid_ivs)
                if min_iv < 0.01:
                    expiry_issues.append(f"Very low implied volatility: {min_iv:.4f}")
                if max_iv > 5.0:
                    expiry_issues.append(f"Very high implied volatility: {max_iv:.4f}")
            
            if expiry_issues:
                issues[expiry] = expiry_issues
        
        return issues
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the market data.
        
        Returns:
            Dictionary with market data summary
        """
        if not self.option_chains:
            return {"status": "empty", "num_expiries": 0}
        
        total_options = sum(len(od.strikes) for od in self.option_chains.values())
        total_valid_ivs = sum(len([iv for iv in od.implied_volatilities if not np.isnan(iv)]) 
                             for od in self.option_chains.values())
        
        # Get range of expiries
        expiries = list(self.option_chains.keys())
        
        # Get range of strikes across all expiries
        all_strikes = []
        for option_data in self.option_chains.values():
            all_strikes.extend(option_data.strikes)
        
        # Get range of implied volatilities
        all_ivs = []
        for option_data in self.option_chains.values():
            all_ivs.extend([iv for iv in option_data.implied_volatilities if not np.isnan(iv)])
        
        return {
            "status": "populated",
            "underlying_symbol": self.underlying_symbol,
            "num_expiries": len(self.option_chains),
            "total_options": total_options,
            "total_valid_ivs": total_valid_ivs,
            "expiry_range": (min(expiries), max(expiries)) if expiries else (None, None),
            "strike_range": (min(all_strikes), max(all_strikes)) if all_strikes else (None, None),
            "iv_range": (min(all_ivs), max(all_ivs)) if all_ivs else (None, None),
            "last_updated": self.last_updated
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "underlying_symbol": self.underlying_symbol,
            "option_chains": {expiry: od.to_dict() for expiry, od in self.option_chains.items()},
            "market_info": self.market_info,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """
        Create MarketData from dictionary.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            MarketData instance
        """
        market_data = cls(data.get("underlying_symbol", ""))
        market_data.market_info = data.get("market_info", {})
        
        if data.get("last_updated"):
            market_data.last_updated = datetime.fromisoformat(data["last_updated"])
        
        for expiry, od_dict in data.get("option_chains", {}).items():
            option_data = OptionData.from_dict(od_dict)
            market_data.add_option_chain(expiry, option_data)
        
        return market_data


def create_sample_market_data() -> MarketData:
    """
    Create sample market data for testing and examples.
    
    Returns:
        Sample MarketData instance with multiple expiries
    """
    market_data = MarketData("SPY")
    
    # Add sample data for different expiries
    # Short-term expiry
    short_term = OptionData(
        strikes=[95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        prices=[10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46],
        spot=100,
        risk_free_rate=0.002,
        time_to_maturity=0.25,  # 3 months
        option_type='call'
    )
    market_data.add_option_chain("2024-03-15", short_term)
    
    # Medium-term expiry
    medium_term = OptionData(
        strikes=[90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110],
        prices=[15.2, 13.8, 12.1, 10.5, 8.9, 7.5, 6.2, 5.1, 4.2, 3.5, 2.9],
        spot=100,
        risk_free_rate=0.002,
        time_to_maturity=0.5,  # 6 months
        option_type='call'
    )
    market_data.add_option_chain("2024-06-21", medium_term)
    
    # Long-term expiry
    long_term = OptionData(
        strikes=[85, 90, 95, 100, 105, 110, 115],
        prices=[20.5, 17.2, 14.1, 11.3, 8.9, 6.8, 5.2],
        spot=100,
        risk_free_rate=0.002,
        time_to_maturity=1.0,  # 1 year
        option_type='call'
    )
    market_data.add_option_chain("2024-12-20", long_term)
    
    return market_data
