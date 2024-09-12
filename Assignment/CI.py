import numpy as np
import pandas as pd

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')


# Function to calculate statistics
def calculate_statistics(data, z=1.96):  # Z = 1.96 for 95% confidence interval
    n = len(data)   
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    std_err = std_dev / np.sqrt(n)
    margin_of_error = z * std_err
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    return mean, std_dev, z, std_err, margin_of_error, ci_lower, ci_upper

# Calculate statistics for screen time and well-being variables
screen_time_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

screen_time_stats_extended = {}
for col in screen_time_columns:
    mean, std_dev, z_stat, std_err, moe, ci_lower, ci_upper = calculate_statistics(dataset2[col].dropna())
    screen_time_stats_extended[col] = (mean, std_dev, z_stat, std_err, moe, ci_lower, ci_upper)

wellbeing_stats_extended = {}
for col in wellbeing_columns:
    mean, std_dev, z_stat, std_err, moe, ci_lower, ci_upper = calculate_statistics(dataset3[col].dropna())
    wellbeing_stats_extended[col] = (mean, std_dev, z_stat, std_err, moe, ci_lower, ci_upper)

# Convert results to DataFrames for display
screen_time_df = pd.DataFrame(screen_time_stats_extended, index=["Mean", "Std Dev", "Z-Statistic", "Standard Error", "Margin of Error", "CI Lower", "CI Upper"]).T
wellbeing_df = pd.DataFrame(wellbeing_stats_extended, index=["Mean", "Std Dev", "Z-Statistic", "Standard Error", "Margin of Error", "CI Lower", "CI Upper"]).T

# Display the results
print("Screen Time Statistics:\n", screen_time_df)
print("\nWell-being Statistics:\n", wellbeing_df)
