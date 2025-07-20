import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.models.svi import SVI
from src.models.gsvi import gSVI
from src.data.market_data import create_sample_market_data
from src.calibration.optimizers import LeastSquaresOptimizer, DifferentialEvolutionOptimizer
from src.calibration.constraints import SVIConstraints
from src.utils.plotting import VolatilitySurfacePlotter, ImpliedVolatilityPlotter
from src.utils.helpers import calculate_calibration_metrics


def calibrate_with_constraints():
    print("CALIBRATION WITH CONSTRAINTS")
    
    # Create sample data
    market_data = create_sample_market_data()
    
    # Get data for one expiry
    option_data = market_data.get_option_chain("2024-06-21")
    valid_data = option_data.get_valid_data()
    
    print(f"Using data with {len(valid_data.strikes)} options")
    print(f"Time to maturity: {valid_data.time_to_maturity} years")
    
    # Create SVI model with constraints
    svi = SVI()
    
    # Set up constraints
    constraints = SVIConstraints()
    
    # Define objective function with constraints
    def constrained_objective(params, log_moneyness, market_ivs, weights=None):
        # Calculate model predictions
        model_ivs = []
        for k in log_moneyness:
            try:
                iv = svi.svi_formula(k, params)
                model_ivs.append(iv)
            except:
                model_ivs.append(np.nan)
        
        model_ivs = np.array(model_ivs)
        
        # Calculate base objective (MSE)
        if weights is not None:
            mse = np.average((model_ivs - np.array(market_ivs))**2, weights=weights)
        else:
            mse = np.mean((model_ivs - np.array(market_ivs))**2)
        
        # Add constraint penalties
        penalty = constraints.total_penalty(params)
        
        return mse + penalty
    
    # Calibrate with different optimizers
    optimizers = {
        'Least Squares': LeastSquaresOptimizer(
            svi, 
            bounds=[(0.001, 1.0), (0.001, 2.0), (-0.999, 0.999), (-2.0, 2.0), (0.001, 2.0)]
        ),
        'Differential Evolution': DifferentialEvolutionOptimizer(
            svi,
            bounds=[(0.001, 1.0), (0.001, 2.0), (-0.999, 0.999), (-2.0, 2.0), (0.001, 2.0)]
        )
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nCalibrating with {name}...")
        
        if isinstance(optimizer, LeastSquaresOptimizer):
            initial_guess = [0.1, 0.3, -0.5, 0.0, 0.2]
            result = optimizer.calibrate(
                data=(valid_data.log_moneyness, valid_data.implied_volatilities),
                objective_function=constrained_objective,
                initial_guess=initial_guess
            )
        else:
            result = optimizer.calibrate(
                data=(valid_data.log_moneyness, valid_data.implied_volatilities),
                objective_function=constrained_objective
            )
        
        print(f"Success: {result.success}")
        print(f"Objective value: {result.objective_value:.6f}")
        
        if result.success:
            # Check constraints
            constraint_summary = constraints.get_violation_summary(result.parameters)
            print(f"All constraints satisfied: {constraint_summary['all_satisfied']}")
            print(f"Total penalty: {constraint_summary['total_penalty']:.6f}")
            
            # Generate predictions
            predictions = svi.predict(valid_data.log_moneyness)
            metrics = calculate_calibration_metrics(predictions, np.array(valid_data.implied_volatilities))
            
            results[name] = {
                'result': result,
                'predictions': predictions,
                'metrics': metrics
            }
            
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"R²: {metrics['r_squared']:.6f}")
    
    return results, valid_data


def analyze_volatility_surface():
    print("VOLATILITY SURFACE ANALYSIS")
    
    # Create market data with multiple expiries
    market_data = create_sample_market_data()
    
    print(f"Market data summary:")
    summary = market_data.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate data
    issues = market_data.validate_data()
    if issues:
        print("\nData validation issues:")
        for expiry, expiry_issues in issues.items():
            print(f"  {expiry}:")
            for issue in expiry_issues:
                print(f"    - {issue}")
    else:
        print("\nNo data validation issues found.")
    
    # Get ATM volatilities
    atm_vols = market_data.get_atm_volatilities()
    print(f"\nATM volatilities:")
    for expiry, vol in atm_vols.items():
        print(f"  {expiry}: {vol:.4f}")
    
    # Create plots
    plotter = VolatilitySurfacePlotter()
    iv_plotter = ImpliedVolatilityPlotter()
    
    # Plot 3D surface
    try:
        fig1 = plotter.plot_volatility_surface_3d(market_data, title="Volatility Surface")
        plt.show()
    except Exception as e:
        print(f"Could not create 3D surface plot: {e}")
    
    # Plot heatmap
    try:
        fig2 = plotter.plot_volatility_surface_heatmap(market_data, title="Volatility Heatmap")
        plt.show()
    except Exception as e:
        print(f"Could not create heatmap: {e}")
    
    # Plot term structure
    try:
        fig3 = iv_plotter.plot_iv_term_structure(market_data, strike_level='atm')
        plt.show()
    except Exception as e:
        print(f"Could not create term structure plot: {e}")
    
    return market_data


def compare_parameterizations():
    print("GSVI PARAMETERIZATION COMPARISON")
    
    # Create sample data
    market_data = create_sample_market_data()
    option_data = market_data.get_option_chain("2024-06-21")
    valid_data = option_data.get_valid_data()
    
    parameterizations = ['raw', 'natural', 'jump_wings']
    results = {}
    
    for param in parameterizations:
        print(f"\nCalibrating gSVI with {param} parameterization...")
        
        gsvi = gSVI(parameterization=param)
        
        try:
            result = gsvi.calibrate(
                log_moneyness=valid_data.log_moneyness,
                implied_volatilities=valid_data.implied_volatilities,
                method='differential_evolution'
            )
            
            if result.success:
                predictions = gsvi.predict(valid_data.log_moneyness)
                metrics = calculate_calibration_metrics(predictions, np.array(valid_data.implied_volatilities))
                
                results[param] = {
                    'result': result,
                    'predictions': predictions,
                    'metrics': metrics
                }
                
                print(f"Success: {result.success}")
                print(f"RMSE: {metrics['rmse']:.6f}")
                print(f"R²: {metrics['r_squared']:.6f}")
                print("Parameters:")
                for name, value in result.get_parameter_dict().items():
                    print(f"  {name}: {value:.6f}")
            else:
                print(f"Calibration failed for {param}")
                
        except Exception as e:
            print(f"Error calibrating {param}: {e}")
    
    # Plot comparison
    if results:
        plotter = VolatilitySurfacePlotter()
        model_predictions = {param: res['predictions'] for param, res in results.items()}
        
        fig = plotter.plot_model_comparison(
            valid_data,
            model_predictions,
            title="gSVI Parameterization Comparison"
        )
        plt.show()
        
        # Print summary table
        print(f"\n{'Parameterization':<15} {'RMSE':<12} {'R²':<12} {'Obj Value':<12}")
        print("-" * 55)
        for param, res in results.items():
            print(f"{param:<15} {res['metrics']['rmse']:<12.6f} "
                  f"{res['metrics']['r_squared']:<12.6f} "
                  f"{res['result'].objective_value:<12.6f}")
    
    return results


def main():
    print("Advanced Calibration Examples")
    
    # Run examples
    try:
        # 1. Calibration with constraints
        constraint_results, constraint_data = calibrate_with_constraints()
        
        # 2. Volatility surface analysis
        surface_data = analyze_volatility_surface()
        
        # 3. Parameterization comparison
        param_results = compare_parameterizations()
        
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
