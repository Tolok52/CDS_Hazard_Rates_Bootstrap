def calculate_prices(new_maturities, fitted_rates):
    prices = []
    for index, swap_rate in zip(new_maturities, fitted_rates):#enumerate(fitted_rates):
        # Adjust the calculation method for different maturities
        if index <= 0.75:  # For maturities 0.25, 0.50, and 0.75
            price = 1 * (1 + swap_rate / 100) ** index
        elif index == 1.00:  # For maturity 1.00
            price = 1 / (1 + swap_rate / 100)
        else:
            sum_previous_prices = sum(prices)
            price = (1 - swap_rate / 100 * sum_previous_prices) / (1 + swap_rate / 100)
        prices.append(price)
    return prices

# Calculate the prices using fitted rates and maturities
calculated_prices = calculate_prices(new_maturities, fitted_rates)

# Displaying the results
results = pd.DataFrame({
    "Maturity": new_maturities,
    "Fitted Swap Rate": fitted_rates,
    "Calculated Price": calculated_prices
})
results
