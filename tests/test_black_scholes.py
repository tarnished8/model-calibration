"""
Unit tests for Black-Scholes model.
"""

import unittest
import numpy as np
from src.models.black_scholes import BlackScholes


class TestBlackScholes(unittest.TestCase):
    """Test cases for BlackScholes class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bs = BlackScholes()
        self.spot = 100.0
        self.strike = 100.0
        self.time_to_maturity = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.2
    
    def test_atm_call_price(self):
        """Test ATM call option pricing."""
        result = self.bs.price(
            self.spot, self.strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        # ATM call should have positive price
        self.assertGreater(result['price'], 0)
        
        # Delta should be around 0.5 for ATM call
        self.assertAlmostEqual(result['delta'], 0.5, delta=0.1)
        
        # Gamma should be positive
        self.assertGreater(result['gamma'], 0)
        
        # Vega should be positive
        self.assertGreater(result['vega'], 0)
    
    def test_atm_put_price(self):
        """Test ATM put option pricing."""
        result = self.bs.price(
            self.spot, self.strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'put'
        )
        
        # ATM put should have positive price
        self.assertGreater(result['price'], 0)
        
        # Delta should be around -0.5 for ATM put
        self.assertAlmostEqual(result['delta'], -0.5, delta=0.1)
        
        # Gamma should be positive (same as call)
        self.assertGreater(result['gamma'], 0)
        
        # Vega should be positive (same as call)
        self.assertGreater(result['vega'], 0)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        call_result = self.bs.price(
            self.spot, self.strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        put_result = self.bs.price(
            self.spot, self.strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'put'
        )
        
        # Put-call parity: C - P = S - K * e^(-r*T)
        parity_diff = (call_result['price'] - put_result['price'] - 
                      self.spot + self.strike * np.exp(-self.risk_free_rate * self.time_to_maturity))
        
        self.assertAlmostEqual(parity_diff, 0, places=10)
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call option."""
        deep_itm_strike = 50.0  # Deep ITM
        
        result = self.bs.price(
            self.spot, deep_itm_strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        # Deep ITM call delta should be close to 1
        self.assertGreater(result['delta'], 0.9)
        
        # Price should be close to intrinsic value
        intrinsic_value = self.spot - deep_itm_strike
        self.assertGreater(result['price'], intrinsic_value)
    
    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option."""
        deep_otm_strike = 150.0  # Deep OTM
        
        result = self.bs.price(
            self.spot, deep_otm_strike, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        # Deep OTM call delta should be close to 0
        self.assertLess(result['delta'], 0.1)
        
        # Price should be small but positive
        self.assertGreater(result['price'], 0)
        self.assertLess(result['price'], 5.0)
    
    def test_zero_volatility(self):
        """Test pricing with zero volatility."""
        result = self.bs.price(
            self.spot, self.strike, self.time_to_maturity,
            self.risk_free_rate, 0.0, 'call'
        )
        
        # With zero volatility, ATM call should have specific value
        expected_price = max(0, self.spot - self.strike * np.exp(-self.risk_free_rate * self.time_to_maturity))
        self.assertAlmostEqual(result['price'], expected_price, places=10)
    
    def test_zero_time_to_maturity(self):
        """Test pricing at expiry."""
        result = self.bs.price(
            self.spot, self.strike, 0.0,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        # At expiry, call price should be max(S-K, 0)
        expected_price = max(0, self.spot - self.strike)
        self.assertAlmostEqual(result['price'], expected_price, places=10)
        
        # Greeks should be zero at expiry
        self.assertEqual(result['gamma'], 0)
        self.assertEqual(result['vega'], 0)
        self.assertEqual(result['theta'], 0)
    
    def test_input_validation(self):
        """Test input validation."""
        # Negative spot price
        with self.assertRaises(ValueError):
            self.bs.price(-100, self.strike, self.time_to_maturity,
                         self.risk_free_rate, self.volatility, 'call')
        
        # Negative strike price
        with self.assertRaises(ValueError):
            self.bs.price(self.spot, -100, self.time_to_maturity,
                         self.risk_free_rate, self.volatility, 'call')
        
        # Negative time to maturity
        with self.assertRaises(ValueError):
            self.bs.price(self.spot, self.strike, -1.0,
                         self.risk_free_rate, self.volatility, 'call')
        
        # Negative volatility
        with self.assertRaises(ValueError):
            self.bs.price(self.spot, self.strike, self.time_to_maturity,
                         self.risk_free_rate, -0.1, 'call')
        
        # Invalid option type
        with self.assertRaises(ValueError):
            self.bs.price(self.spot, self.strike, self.time_to_maturity,
                         self.risk_free_rate, self.volatility, 'invalid')
    
    def test_multiple_options(self):
        """Test pricing multiple options at once."""
        strikes = [90, 95, 100, 105, 110]
        
        results = self.bs.price_multiple(
            self.spot, strikes, self.time_to_maturity,
            self.risk_free_rate, self.volatility, 'call'
        )
        
        self.assertEqual(len(results), len(strikes))
        
        # Call prices should be decreasing with strike
        prices = [result['price'] for result in results]
        for i in range(len(prices) - 1):
            self.assertGreater(prices[i], prices[i + 1])
    
    def test_greeks_consistency(self):
        """Test Greeks consistency using finite differences."""
        # Test delta using finite difference
        h = 0.01
        price_up = self.bs.price(self.spot + h, self.strike, self.time_to_maturity,
                                self.risk_free_rate, self.volatility, 'call')['price']
        price_down = self.bs.price(self.spot - h, self.strike, self.time_to_maturity,
                                  self.risk_free_rate, self.volatility, 'call')['price']
        
        delta_fd = (price_up - price_down) / (2 * h)
        
        result = self.bs.price(self.spot, self.strike, self.time_to_maturity,
                              self.risk_free_rate, self.volatility, 'call')
        
        self.assertAlmostEqual(result['delta'], delta_fd, places=3)
        
        # Test vega using finite difference
        h_vol = 0.001
        price_vol_up = self.bs.price(self.spot, self.strike, self.time_to_maturity,
                                    self.risk_free_rate, self.volatility + h_vol, 'call')['price']
        price_vol_down = self.bs.price(self.spot, self.strike, self.time_to_maturity,
                                      self.risk_free_rate, self.volatility - h_vol, 'call')['price']
        
        vega_fd = (price_vol_up - price_vol_down) / (2 * h_vol)
        
        # Vega is reported as percentage, so multiply by 100
        self.assertAlmostEqual(result['vega'] * 100, vega_fd, places=2)


if __name__ == '__main__':
    unittest.main()
