import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge dataset1 (demographics) with dataset2 (screen time) based on the 'ID' column
merged_data = pd.merge(dataset1[['ID', 'gender']], dataset2, on='ID')

# Now merge the result with dataset3 (well-being) based on the 'ID' column
total_merged_data = pd.merge(merged_data, dataset3, on='ID')

# Define the well-being columns
wellbeing_columns_extended = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 
                              'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate total screen time for weekends and weekdays
total_merged_data['total_screen_time_we'] = total_merged_data['C_we'] + total_merged_data['G_we'] + total_merged_data['S_we'] + total_merged_data['T_we']
total_merged_data['total_screen_time_wk'] = total_merged_data['C_wk'] + total_merged_data['G_wk'] + total_merged_data['S_wk'] + total_merged_data['T_wk']

# Hypothesis test results storage
results_weekend = {}
results_weekday = {}

# Hypothesis testing for weekends
for col in wellbeing_columns_extended:
    group_more_than_3_hours_we = total_merged_data[total_merged_data['total_screen_time_we'] > 3][col].dropna()
    group_less_than_equal_3_hours_we = total_merged_data[total_merged_data['total_screen_time_we'] <= 3][col].dropna()
    t_stat_we, p_value_we = stats.ttest_ind(group_more_than_3_hours_we, group_less_than_equal_3_hours_we, equal_var=False)
    results_weekend[col] = (t_stat_we, p_value_we)

# Hypothesis testing for weekdays
for col in wellbeing_columns_extended:
    group_more_than_3_hours_wk = total_merged_data[total_merged_data['total_screen_time_wk'] > 3][col].dropna()
    group_less_than_equal_3_hours_wk = total_merged_data[total_merged_data['total_screen_time_wk'] <= 3][col].dropna()
    t_stat_wk, p_value_wk = stats.ttest_ind(group_more_than_3_hours_wk, group_less_than_equal_3_hours_wk, equal_var=False)
    results_weekday[col ] = (t_stat_wk, p_value_wk)
    
def hypothesis_decision(p_value, alpha=0.05):
    """
    Returns whether to reject or accept the null hypothesis based on the p-value and significance level.
    """
    if p_value < alpha:
        return "Reject Null Hypothesis"
    else:
        return "Fail to Reject Null Hypothesis (Accept Null Hypothesis)"

# Convert the results to DataFrame for better visualization
weekend_results_df = pd.DataFrame(results_weekend, index=["T-Statistic", "P-Value"]).T
weekday_results_df = pd.DataFrame(results_weekday, index=["T-Statistic", "P-Value"]).T

# Apply the decision to both weekend and weekday hypothesis test results
weekend_results_df['Decision'] = weekend_results_df['P-Value'].apply(hypothesis_decision)
weekday_results_df['Decision'] = weekday_results_df['P-Value'].apply(hypothesis_decision)

# Print or display the results
print("Hypothesis Test Results - Weekends:\n", weekend_results_df)
print("\nHypothesis Test Results - Weekdays:\n", weekday_results_df)
