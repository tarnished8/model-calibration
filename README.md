# Model Calibration Framework

This is my Model Calibration project, which includes the features stated below, and was transformed to a modular repository from the initial Jupyter notebook.

## Features

- **Black-Scholes Option Pricing**: Complete implementation with Greeks calculation
- **Implied Volatility Calculation**: Newton-Raphson method for extracting implied volatilities
- **SVI Model**: Stochastic Volatility Inspired model for volatility surface fitting
- **gSVI Model**: Generalized SVI with advanced calibration capabilities
- **Parameter Optimization**: Multiple optimization methods including least squares and differential evolution
- **Visualization**: Plotting tools for volatility surfaces and model analysis

## Project Structure

```
model_calibration/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── black_scholes.py      # Black-Scholes pricing and Greeks
│   │   ├── implied_volatility.py # Implied volatility calculation
│   │   ├── svi.py                # SVI model implementation
│   │   └── gsvi.py               # Generalized SVI model
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── base.py               # Base calibration classes
│   │   ├── optimizers.py         # Optimization algorithms
│   │   └── constraints.py        # Parameter constraints
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_data.py        # Market data handling
│   │   └── option_data.py        # Option data structures
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py           # Visualization utilities
│       └── helpers.py            # Helper functions
├── examples/
│   ├── basic_usage.py            # Basic usage examples
│   ├── svi_calibration.py        # SVI calibration example
│   └── volatility_surface.py    # Volatility surface analysis
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py
│   ├── test_svi.py
│   └── test_calibration.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/model-calibration.git
cd model-calibration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from src.models.black_scholes import BlackScholes
from src.models.implied_volatility import ImpliedVolatility
from src.models.svi import SVI
from src.data.market_data import OptionData

# Create option data
option_data = OptionData(
    strikes=[95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
    prices=[10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46],
    spot=100,
    risk_free_rate=0.002,
    time_to_maturity=1.0
)

# Calculate Black-Scholes price and Greeks
bs = BlackScholes()
result = bs.price(spot=100, strike=110, time_to_maturity=1, 
                  risk_free_rate=0.002, volatility=0.2)
print(f"Price: {result['price']:.4f}")
print(f"Delta: {result['delta']:.4f}")

# Calculate implied volatility
iv_calc = ImpliedVolatility()
implied_vol = iv_calc.calculate(spot=100, strike=110, time_to_maturity=1,
                               risk_free_rate=0.002, market_price=6.0)
print(f"Implied Volatility: {implied_vol:.4f}")

# Fit SVI model
svi = SVI()
svi.calibrate(option_data)
print(f"SVI Parameters: {svi.parameters}")
```

## Dependencies

- numpy
- scipy
- matplotlib
- pandas (optional, for data handling)

