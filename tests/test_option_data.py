"""
Unit tests for OptionData class.
"""

import unittest
import numpy as np
from src.data.option_data import OptionData, create_sample_data


class TestOptionData(unittest.TestCase):
    """Test cases for OptionData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strikes = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]
        self.prices = [10.5, 9.2, 8.1, 7.2, 6.5, 5.9, 5.4, 5.0, 4.7, 4.5, 4.3]
        self.spot = 100.0
        self.risk_free_rate = 0.05
        self.time_to_maturity = 0.25
        self.option_type = 'call'
    
    def test_basic_creation(self):
        """Test basic OptionData creation."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        self.assertEqual(len(option_data.strikes), len(self.strikes))
        self.assertEqual(len(option_data.prices), len(self.prices))
        self.assertEqual(option_data.spot, self.spot)
        self.assertEqual(option_data.risk_free_rate, self.risk_free_rate)
        self.assertEqual(option_data.time_to_maturity, self.time_to_maturity)
        self.assertEqual(option_data.option_type, self.option_type)
    
    def test_derived_quantities_calculation(self):
        """Test that derived quantities are calculated correctly."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        # Check log-moneyness calculation
        self.assertIsNotNone(option_data.log_moneyness)
        self.assertEqual(len(option_data.log_moneyness), len(self.strikes))
        
        # ATM option should have log-moneyness close to 0
        atm_index = self.strikes.index(100)
        self.assertAlmostEqual(option_data.log_moneyness[atm_index], 0.0, places=10)
        
        # Check implied volatilities calculation
        self.assertIsNotNone(option_data.implied_volatilities)
        self.assertEqual(len(option_data.implied_volatilities), len(self.strikes))
        
        # Most IVs should be valid (not NaN)
        valid_ivs = [iv for iv in option_data.implied_volatilities if not np.isnan(iv)]
        self.assertGreater(len(valid_ivs), len(self.strikes) * 0.5)  # At least 50% should be valid
    
    def test_input_validation(self):
        """Test input validation."""
        # Mismatched lengths
        with self.assertRaises(ValueError):
            OptionData(
                strikes=[95, 100, 105],
                prices=[5.0, 6.0],  # Wrong length
                spot=self.spot
            )
        
        # Empty data
        with self.assertRaises(ValueError):
            OptionData(
                strikes=[],
                prices=[],
                spot=self.spot
            )
        
        # Negative spot
        with self.assertRaises(ValueError):
            OptionData(
                strikes=self.strikes,
                prices=self.prices,
                spot=-100.0
            )
        
        # Negative time to maturity
        with self.assertRaises(ValueError):
            OptionData(
                strikes=self.strikes,
                prices=self.prices,
                spot=self.spot,
                time_to_maturity=-0.25
            )
        
        # Negative strikes
        with self.assertRaises(ValueError):
            OptionData(
                strikes=[-95, 100, 105],
                prices=[5.0, 6.0, 7.0],
                spot=self.spot
            )
        
        # Negative prices
        with self.assertRaises(ValueError):
            OptionData(
                strikes=[95, 100, 105],
                prices=[-5.0, 6.0, 7.0],
                spot=self.spot
            )
        
        # Invalid option type
        with self.assertRaises(ValueError):
            OptionData(
                strikes=self.strikes,
                prices=self.prices,
                spot=self.spot,
                option_type='invalid'
            )
    
    def test_get_valid_data(self):
        """Test filtering to valid data only."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        valid_data = option_data.get_valid_data()
        
        # Valid data should have no NaN IVs
        for iv in valid_data.implied_volatilities:
            self.assertFalse(np.isnan(iv))
        
        # Valid data should have fewer or equal points
        self.assertLessEqual(len(valid_data.strikes), len(option_data.strikes))
    
    def test_filter_by_moneyness(self):
        """Test filtering by moneyness range."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        # Filter to narrow range around ATM
        filtered_data = option_data.filter_by_moneyness(-0.1, 0.1)
        
        # Should have fewer points
        self.assertLess(len(filtered_data.strikes), len(option_data.strikes))
        
        # All log-moneyness values should be in range
        for lm in filtered_data.log_moneyness:
            self.assertGreaterEqual(lm, -0.1)
            self.assertLessEqual(lm, 0.1)
    
    def test_add_weights(self):
        """Test weight generation."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        # Test equal weights
        equal_weights = option_data.add_weights('equal')
        self.assertEqual(len(equal_weights), len(option_data.strikes))
        self.assertTrue(all(w == 1.0 for w in equal_weights))
        
        # Test vega weights
        vega_weights = option_data.add_weights('vega')
        self.assertEqual(len(vega_weights), len(option_data.strikes))
        self.assertTrue(all(w >= 0 for w in vega_weights))
        
        # Test price weights
        price_weights = option_data.add_weights('price')
        self.assertEqual(len(price_weights), len(option_data.strikes))
        self.assertTrue(all(w >= 0 for w in price_weights))
        
        # Test invalid weight type
        with self.assertRaises(ValueError):
            option_data.add_weights('invalid')
    
    def test_summary(self):
        """Test data summary generation."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        summary = option_data.summary()
        
        # Check required fields
        required_fields = [
            'num_options', 'num_valid_ivs', 'spot', 'risk_free_rate',
            'time_to_maturity', 'option_type', 'strike_range', 'price_range',
            'moneyness_range', 'iv_range', 'iv_mean', 'iv_std'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check values make sense
        self.assertEqual(summary['num_options'], len(self.strikes))
        self.assertEqual(summary['spot'], self.spot)
        self.assertEqual(summary['strike_range'], (min(self.strikes), max(self.strikes)))
        self.assertEqual(summary['price_range'], (min(self.prices), max(self.prices)))
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        option_data = OptionData(
            strikes=self.strikes,
            prices=self.prices,
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            time_to_maturity=self.time_to_maturity,
            option_type=self.option_type
        )
        
        # Convert to dict
        data_dict = option_data.to_dict()
        
        # Check all fields are present
        required_fields = [
            'strikes', 'prices', 'spot', 'risk_free_rate',
            'time_to_maturity', 'option_type', 'implied_volatilities', 'log_moneyness'
        ]
        
        for field in required_fields:
            self.assertIn(field, data_dict)
        
        # Convert back from dict
        reconstructed = OptionData.from_dict(data_dict)
        
        # Check that data is preserved
        self.assertEqual(reconstructed.strikes, option_data.strikes)
        self.assertEqual(reconstructed.prices, option_data.prices)
        self.assertEqual(reconstructed.spot, option_data.spot)
        self.assertEqual(reconstructed.risk_free_rate, option_data.risk_free_rate)
        self.assertEqual(reconstructed.time_to_maturity, option_data.time_to_maturity)
        self.assertEqual(reconstructed.option_type, option_data.option_type)
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        sample_data = create_sample_data()
        
        # Should be valid OptionData
        self.assertIsInstance(sample_data, OptionData)
        
        # Should have reasonable number of options
        self.assertGreater(len(sample_data.strikes), 5)
        
        # Should have valid derived quantities
        self.assertIsNotNone(sample_data.log_moneyness)
        self.assertIsNotNone(sample_data.implied_volatilities)
        
        # Most IVs should be valid
        valid_ivs = [iv for iv in sample_data.implied_volatilities if not np.isnan(iv)]
        self.assertGreater(len(valid_ivs), 0)


if __name__ == '__main__':
    unittest.main()
