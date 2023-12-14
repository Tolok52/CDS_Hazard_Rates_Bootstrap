import numpy as np
from scipy.optimize import minimize

# Given data
maturities = np.array([12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 180, 240, 300, 360]) / 12
swap_rates = np.array([3.239, 2.833, 2.625, 2.463, 2.405, 2.355, 2.323, 2.313, 2.319, 2.378, 2.535, 2.517, 2.436, 2.444])

# Define the NSS model function with two taos
def nss_model(params, t):
    b0, b1, b2, b3, tao1, tao2 = params
    return b0 + b1 * ((1 - np.exp(-t / tao1)) / (t / tao1)) + b2 * (((1 - np.exp(-t / tao1)) / (t / tao1)) - np.exp(-t / tao1)) + b3 * (((1 - np.exp(-t / tao2)) / (t / tao2)) - np.exp(-t / tao2) - 0.5 * (t / tao2) * np.exp(-t / tao2))

# Define the objective function to minimize
def objective_function(params):
    b0, b1, b2, b3, tao1, tao2 = params
    # Add penalties for violating constraints
    penalty1 = max(0, -b1 - b2) ** 2
    penalty2 = max(0, -b0) ** 2
    penalty3 = max(0, -tao1) ** 2
    penalty4 = max(0, -tao2) ** 2
    predicted_rates = nss_model(params, maturities)
    return np.sum((predicted_rates - swap_rates) ** 2) + penalty1 + penalty2 + penalty3 + penalty4

# Initial guess for the parameters with two taos
initial_params = [2.5, 1, -0.5, -0.5, 5, 10]  # You can adjust the initial values as needed

# Minimize the objective function
result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=((0, None), (None, None), (None, None), (None, None), (0, None), (0, None)))

# Fitted parameters
fitted_params = result.x
