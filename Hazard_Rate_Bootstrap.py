import numpy as np
import scipy.optimize as optim
import scipy.interpolate as polate

# Define the CDS_bootstrap function
def CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R):
    # Checks
    if len(cds_spreads) != len(cds_tenor):
        print("CDS spread array does not match CDS tenor array.")
        return None

    if len(yield_curve) != len(yield_tenor):
        print("Yield curve array does not match yield tenor.")
        return None

    # Interpolation/Extrapolation function
    interp = polate.interp1d(yield_tenor, yield_curve, 'linear', fill_value='extrapolate')

    # The bootstrap function
    def bootstrap(h, given_haz, s, cds_tenor, yield_curve, prem_per_year, R):
        a = 1 / prem_per_year
        maturities = [0] + list(cds_tenor)
        pmnt = 0
        dflt = 0
        auc = 0

        # Calculate value of payments for given hazard rate curve values
        for i in range(1, len(maturities) - 1):
            num_points = int((maturities[i] - maturities[i - 1]) * prem_per_year + 1)
            t = np.linspace(maturities[i - 1], maturities[i], num_points)
            r = interp(t)

            for j in range(1, len(t)):
                surv_prob_prev = np.exp(-given_haz[i - 1] * (t[j - 1] - t[0]) - auc)
                surv_prob_curr = np.exp(-given_haz[i - 1] * (t[j] - t[0]) - auc)
                pmnt += s * a * np.exp(-r[j] * t[j]) * 0.5 * (surv_prob_prev + surv_prob_curr)
                dflt += np.exp(-r[j] * t[j]) * (1 - R) * (surv_prob_prev - surv_prob_curr)

            auc += (t[-1] - t[0]) * given_haz[i - 1]

        # Set up calculations for payments with the unknown hazard rate value
        num_points = int((maturities[-1] - maturities[-2]) * prem_per_year + 1)
        t = np.linspace(maturities[-2], maturities[-1], num_points)
        r = interp(t)

        for i in range(1, len(t)):
            surv_prob_prev = np.exp(-h * (t[i - 1] - t[0]) - auc)
            surv_prob_curr = np.exp(-h * (t[i] - t[0]) - auc)
            pmnt += s * a * np.exp(-r[i] * t[i]) * 0.5 * (surv_prob_prev + surv_prob_curr)
            dflt += np.exp(-r[i] * t[i]) * (1 - R) * (surv_prob_prev - surv_prob_curr)

        return abs(pmnt - dflt)

    haz_rates = []
    cumulative_hazard = 0.0
    surv_prob = [1.0]  # Initial survival probability is 1 at time 0
    t = [0] + list(cds_tenor)

    for i in range(len(cds_spreads)):
        get_haz = lambda x: bootstrap(x, haz_rates, cds_spreads[i], cds_tenor[:i + 1], yield_curve[:i + 1], prem_per_year, R)
        haz = round(optim.minimize(get_haz, cds_spreads[i] / (1 - R), method='SLSQP', tol=1e-10).x[0], 8)
        haz_rates.append(haz)
        cumulative_hazard += haz * (cds_tenor[i] - (cds_tenor[i - 1] if i > 0 else 0))
        surv_prob.append(np.exp(-cumulative_hazard))

    return haz_rates, surv_prob

# Sample input data
cds_spreads = np.array([24.130, 30.280, 40.800, 51.230, 61.900, 72.590, 85.030, 95.670, 106.920, 115.570]) / 10000  # Convert basis points to decimal
cds_tenor = np.array([0.5, 1, 2, 3, 4, 5, 7, 10, 20, 30])
yield_curve = np.array([0.0247, 0.0265, 0.0308, 0.0323, 0.0504, 0.0613, 0.0682, 0.0726, 0.0756, 0.0775, 0.0789, 0.0797, 0.0803, 0.0808, 0.0811, 0.0813, 0.0815, 0.0816, 0.0818, 0.0820, 0.0822, 0.0824, 0.0827, 0.0830, 0.0833, 0.0836, 0.0840, 0.0844, 0.0848, 0.0852, 0.0857, 0.0862, 0.0867, 0.0872, 0.0877, 0.0882, 0.0888, 0.0893, 0.0899, 0.0905, 0.0910, 0.0916, 0.0922, 0.0928, 0.0933, 0.0939, 0.0945, 0.0950, 0.0956, 0.0961, 0.0967, 0.0972, 0.0977, 0.0982, 0.0987, 0.0992, 0.0996, 0.1001, 0.1005, 0.1010, 0.1014, 0.1018, 0.1021, 0.1025, 0.1028, 0.1031, 0.1034, 0.1037, 0.1039, 0.1042, 0.1044, 0.1045, 0.1047, 0.1048, 0.1049, 0.1050, 0.1051, 0.1051, 0.1051, 0.1051, 0.1051, 0.1050, 0.1049, 0.1048, 0.1047, 0.1045, 0.1043, 0.1041, 0.1039, 0.1036, 0.1033, 0.1030, 0.1027, 0.1024, 0.1020, 0.1016, 0.1012, 0.1008, 0.1003, 0.0998, 0.0994, 0.0989, 0.0983, 0.0978, 0.0973, 0.0967, 0.0961, 0.0956, 0.0950, 0.0944, 0.0938, 0.0932, 0.0925, 0.0919, 0.0913, 0.0906, 0.0900, 0.0894, 0.0887, 0.0881]) # Convert to decimal
yield_tenor = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0])
prem_per_year = 4  # Quarterly payments
R = 0.4  # Recovery rate




# Calculate hazard rates and survival probabilities
hazard_rates, survival_probabilities = CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R)
survival_probabilities = survival_probabilities[1:]
# Print results
print("Hazard Rates:", hazard_rates)
print("Survival Probabilities:", survival_probabilities)
