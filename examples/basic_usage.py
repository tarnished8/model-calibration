"""
Basic usage example for the model calibration package.

This script demonstrates the basic functionality of the package including:
- Loading option data
- Calculating implied volatilities
- Calibrating SVI and gSVI models
- Plotting results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.black_scholes import BlackScholes
from src.models.implied_volatility import ImpliedVolatility
from src.models.svi import SVI
from src.models.gsvi import gSVI
from src.data.option_data import OptionData, create_sample_data
from src.utils.plotting import VolatilitySurfacePlotter
from src.utils.helpers import calculate_calibration_metrics


def main():
    print("Model Calibration Package - Basic Usage Example")
    
    # 1. Create sample option data
    print("\n1. Creating sample option data...")
    option_data = create_sample_data()
    print(f"Created data with {len(option_data.strikes)} options")
    print(f"Spot price: {option_data.spot}")
    print(f"Time to maturity: {option_data.time_to_maturity} years")
    
    # Display data summary
    summary = option_data.summary()
    print(f"Strike range: {summary['strike_range']}")
    print(f"IV range: {summary['iv_range']}")
    print(f"Valid IVs: {summary['num_valid_ivs']}/{summary['num_options']}")
    
    # 2. Black-Scholes pricing example
    print("\n2. Black-Scholes pricing example:")
    bs = BlackScholes()
    
    # Price a single option
    result = bs.price(
        spot=100,
        strike=100,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option_type='call'
    )
    
    print(f"ATM Call price: {result['price']:.4f}")
    print(f"Delta: {result['delta']:.4f}")
    print(f"Gamma: {result['gamma']:.4f}")
    print(f"Vega: {result['vega']:.4f}")
    
    # 3. Implied volatility calculation
    print("\n3. Implied volatility calculation...")
    iv_calc = ImpliedVolatility()
    
    # Calculate IV for a single option
    market_price = 5.0
    implied_vol = iv_calc.calculate(
        spot=100,
        strike=105,
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        market_price=market_price,
        option_type='call'
    )
    
    print(f"Market price: {market_price}")
    print(f"Implied volatility: {implied_vol:.4f}")
    
    # 4. SVI model calibration
    print("\n4. SVI model calibration...")
    
    # Get valid data for calibration
    valid_data = option_data.get_valid_data()
    
    # Create SVI model
    svi = SVI()
    
    # Calibrate using least squares
    print("Calibrating SVI model...")
    svi_result = svi.calibrate(
        log_moneyness=valid_data.log_moneyness,
        implied_volatilities=valid_data.implied_volatilities,
        method='least_squares'
    )
    
    print(f"Calibration successful: {svi_result.success}")
    print(f"Final objective value: {svi_result.objective_value:.6f}")
    print("SVI parameters:")
    for name, value in svi_result.get_parameter_dict().items():
        print(f"  {name}: {value:.6f}")
    
    # Generate predictions
    svi_predictions = svi.predict(valid_data.log_moneyness)
    
    # Calculate metrics
    svi_metrics = calculate_calibration_metrics(
        svi_predictions, 
        np.array(valid_data.implied_volatilities)
    )
    print(f"SVI RMSE: {svi_metrics['rmse']:.6f}")
    print(f"SVI R²: {svi_metrics['r_squared']:.6f}")
    
    # 5. gSVI model calibration
    print("\n5. gSVI model calibration...")
    
    # Create gSVI model with natural parameterization
    gsvi = gSVI(parameterization='natural')
    
    # Calibrate
    print("Calibrating gSVI model...")
    gsvi_result = gsvi.calibrate(
        log_moneyness=valid_data.log_moneyness,
        implied_volatilities=valid_data.implied_volatilities,
        method='differential_evolution'
    )
    
    print(f"Calibration successful: {gsvi_result.success}")
    print(f"Final objective value: {gsvi_result.objective_value:.6f}")
    print("gSVI parameters:")
    for name, value in gsvi_result.get_parameter_dict().items():
        print(f"  {name}: {value:.6f}")
    
    # Generate predictions
    gsvi_predictions = gsvi.predict(valid_data.log_moneyness)
    
    # Calculate metrics
    gsvi_metrics = calculate_calibration_metrics(
        gsvi_predictions, 
        np.array(valid_data.implied_volatilities)
    )
    print(f"gSVI RMSE: {gsvi_metrics['rmse']:.6f}")
    print(f"gSVI R²: {gsvi_metrics['r_squared']:.6f}")
    
    # 6. Plotting results
    print("\n6. Creating plots...")
    
    plotter = VolatilitySurfacePlotter()
    
    # Plot SVI fit
    fig1 = plotter.plot_implied_volatility_smile(
        valid_data,
        svi_predictions,
        model_name="SVI",
        title="SVI Model Calibration"
    )
    plt.show()
    
    # Plot model comparison
    model_results = {
        'SVI': svi_predictions,
        'gSVI': gsvi_predictions
    }
    
    fig2 = plotter.plot_model_comparison(
        valid_data,
        model_results,
        title="Model Comparison: SVI vs gSVI"
    )
    plt.show()
    
    # 7. Summary
    print("\n7. Summary")
    print("=" * 40)
    print(f"{'Model':<10} {'RMSE':<12} {'R²':<12}")
    print("-" * 40)
    print(f"{'SVI':<10} {svi_metrics['rmse']:<12.6f} {svi_metrics['r_squared']:<12.6f}")
    print(f"{'gSVI':<10} {gsvi_metrics['rmse']:<12.6f} {gsvi_metrics['r_squared']:<12.6f}")
    
    print("\nExample completed successfully!")
    print("Check the plots to see the model fits.")


if __name__ == "__main__":
    main()
