"""
Unit tests for SVI model.
"""

import unittest
import numpy as np
from src.models.svi import SVI


class TestSVI(unittest.TestCase):
    """Test cases for SVI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.svi = SVI()
        
        # Valid SVI parameters
        self.valid_params = [0.1, 0.3, -0.5, 0.0, 0.2]  # a, b, rho, m, sigma
        
        # Sample data for calibration
        self.log_moneyness = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        self.implied_volatilities = [0.25, 0.22, 0.20, 0.19, 0.20, 0.22, 0.25]
    
    def test_svi_formula(self):
        """Test SVI formula calculation."""
        k = 0.0  # ATM
        result = self.svi.svi_formula(k, self.valid_params)
        
        # Result should be positive
        self.assertGreater(result, 0)
        
        # For ATM (k=0), formula simplifies
        a, b, rho, m, sigma = self.valid_params
        expected = np.sqrt(a + b * (rho * (-m) + np.sqrt(m**2 + sigma**2)))
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_svi_formula_multiple_points(self):
        """Test SVI formula for multiple points."""
        results = []
        for k in self.log_moneyness:
            result = self.svi.svi_formula(k, self.valid_params)
            results.append(result)
            self.assertGreater(result, 0)  # All results should be positive
        
        # Results should form a reasonable volatility smile
        self.assertEqual(len(results), len(self.log_moneyness))
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters should pass
        self.assertTrue(self.svi._validate_parameters(self.valid_params))
        
        # Invalid parameters should fail
        invalid_params = [
            [-0.1, 0.3, -0.5, 0.0, 0.2],  # negative a
            [0.1, -0.3, -0.5, 0.0, 0.2],  # negative b
            [0.1, 0.3, -1.5, 0.0, 0.2],   # |rho| > 1
            [0.1, 0.3, -0.5, 0.0, -0.2],  # negative sigma
        ]
        
        for params in invalid_params:
            self.assertFalse(self.svi._validate_parameters(params))
    
    def test_calibration_least_squares(self):
        """Test calibration using least squares."""
        result = self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            method='least_squares'
        )
        
        # Calibration should succeed
        self.assertTrue(result.success)
        
        # Parameters should be valid
        self.assertTrue(self.svi._validate_parameters(result.parameters))
        
        # Model should be calibrated
        self.assertTrue(self.svi.is_calibrated())
        
        # Objective value should be reasonable
        self.assertGreater(result.objective_value, 0)
        self.assertLess(result.objective_value, 1.0)  # Should be small for good fit
    
    def test_calibration_differential_evolution(self):
        """Test calibration using differential evolution."""
        result = self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            method='differential_evolution'
        )
        
        # Calibration should succeed
        self.assertTrue(result.success)
        
        # Parameters should be valid
        self.assertTrue(self.svi._validate_parameters(result.parameters))
        
        # Model should be calibrated
        self.assertTrue(self.svi.is_calibrated())
    
    def test_prediction(self):
        """Test model prediction."""
        # First calibrate the model
        self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            method='least_squares'
        )
        
        # Test prediction for single point
        prediction = self.svi.predict(0.0)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)
        
        # Test prediction for multiple points
        predictions = self.svi.predict(self.log_moneyness)
        self.assertEqual(len(predictions), len(self.log_moneyness))
        
        for pred in predictions:
            self.assertGreater(pred, 0)
    
    def test_prediction_without_calibration(self):
        """Test that prediction fails without calibration."""
        with self.assertRaises(ValueError):
            self.svi.predict(0.0)
    
    def test_get_parameter_names(self):
        """Test parameter names."""
        names = self.svi.get_parameter_names()
        expected_names = ['a', 'b', 'rho', 'm', 'sigma']
        self.assertEqual(names, expected_names)
    
    def test_calibration_with_weights(self):
        """Test calibration with weights."""
        weights = [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]  # Higher weight for ATM
        
        result = self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            weights=weights,
            method='least_squares'
        )
        
        self.assertTrue(result.success)
        self.assertTrue(self.svi._validate_parameters(result.parameters))
    
    def test_calibration_with_bad_data(self):
        """Test calibration with problematic data."""
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.svi.calibrate(
                log_moneyness=[0.0, 0.1],
                implied_volatilities=[0.2, 0.21, 0.22],
                method='least_squares'
            )
        
        # Test with NaN values
        bad_ivs = [0.25, np.nan, 0.20, 0.19, 0.20, 0.22, 0.25]
        with self.assertRaises(ValueError):
            self.svi.calibrate(
                log_moneyness=self.log_moneyness,
                implied_volatilities=bad_ivs,
                method='least_squares'
            )
        
        # Test with negative volatilities
        negative_ivs = [0.25, -0.22, 0.20, 0.19, 0.20, 0.22, 0.25]
        with self.assertRaises(ValueError):
            self.svi.calibrate(
                log_moneyness=self.log_moneyness,
                implied_volatilities=negative_ivs,
                method='least_squares'
            )
    
    def test_no_arbitrage_constraints(self):
        """Test that calibrated parameters satisfy no-arbitrage constraints."""
        result = self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            method='least_squares'
        )
        
        if result.success:
            a, b, rho, m, sigma = result.parameters
            
            # Basic no-arbitrage conditions
            self.assertGreater(a, 0)
            self.assertGreater(b, 0)
            self.assertGreaterEqual(abs(rho), 0)
            self.assertLess(abs(rho), 1)
            self.assertGreater(sigma, 0)
            
            # Additional no-arbitrage constraints
            self.assertGreaterEqual(a + b * sigma * (1 + abs(rho)), 0)
            self.assertGreaterEqual(b * (1 + rho), 0)
            self.assertGreaterEqual(b * (1 - rho), 0)
    
    def test_smile_shape(self):
        """Test that calibrated model produces reasonable smile shape."""
        result = self.svi.calibrate(
            log_moneyness=self.log_moneyness,
            implied_volatilities=self.implied_volatilities,
            method='least_squares'
        )
        
        if result.success:
            # Generate predictions for a wider range
            test_moneyness = np.linspace(-0.5, 0.5, 21)
            predictions = self.svi.predict(test_moneyness.tolist())
            
            # Find minimum (should be near ATM for typical smile)
            min_idx = np.argmin(predictions)
            min_moneyness = test_moneyness[min_idx]
            
            # Minimum should be reasonably close to ATM
            self.assertLess(abs(min_moneyness), 0.2)
            
            # Volatilities should increase away from minimum (smile shape)
            left_wing = predictions[:min_idx]
            right_wing = predictions[min_idx+1:]
            
            if len(left_wing) > 1:
                # Left wing should be increasing (going towards ATM)
                for i in range(len(left_wing) - 1):
                    self.assertGreaterEqual(left_wing[i+1], left_wing[i] - 0.01)  # Allow small tolerance
            
            if len(right_wing) > 1:
                # Right wing should be increasing (going away from ATM)
                for i in range(len(right_wing) - 1):
                    self.assertGreaterEqual(right_wing[i+1], right_wing[i] - 0.01)  # Allow small tolerance


if __name__ == '__main__':
    unittest.main()
