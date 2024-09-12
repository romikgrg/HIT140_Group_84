import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge dataset1 with dataset2 based on the ID
merged_data = pd.merge(dataset1[['ID', 'gender']], dataset2, on='ID')

# Now merge the result with dataset3 based on the ID
total_merged_data = pd.merge(merged_data, dataset3, on='ID')

# Define the well-being columns
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 
                              'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate total screen time for weekends and weekdays
total_merged_data['total_weekend'] = total_merged_data['C_we'] + total_merged_data['G_we'] + total_merged_data['S_we'] + total_merged_data['T_we']
total_merged_data['total_weekdays'] = total_merged_data['C_wk'] + total_merged_data['G_wk'] + total_merged_data['S_wk'] + total_merged_data['T_wk']

# Hypothesis test results storage
results_weekend = {}
results_weekday = {}

# Hypothesis testing for weekends
for col in wellbeing_columns:
    group_more_than_3_hours_we = total_merged_data[total_merged_data['total_weekend'] > 3][col].dropna()
    group_less_than_equal_3_hours_we = total_merged_data[total_merged_data['total_weekend'] <= 3][col].dropna()
    t_stat_we, p_value_we = stats.ttest_ind(group_more_than_3_hours_we, group_less_than_equal_3_hours_we, equal_var=False)
    results_weekend[col] = (t_stat_we, p_value_we)

# Hypothesis testing for weekdays
for col in wellbeing_columns:
    group_more_than_3_hours_wk = total_merged_data[total_merged_data['total_weekdays'] > 3][col].dropna()
    group_less_than_equal_3_hours_wk = total_merged_data[total_merged_data['total_weekdays'] <= 3][col].dropna()
    t_stat_wk, p_value_wk = stats.ttest_ind(group_more_than_3_hours_wk, group_less_than_equal_3_hours_wk, equal_var=False)
    results_weekday[col ] = (t_stat_wk, p_value_wk)
    
def hypothesis_decision(p_value):
    
    if p_value < 0.05:
        return "Reject Null Hypothesis"
    else:
        return "Fail to Reject Null Hypothesis"

# Convert the results to DataFrame
weekend_results_df = pd.DataFrame(results_weekend, index=["T-Statistic", "P-Value"]).T
weekday_results_df = pd.DataFrame(results_weekday, index=["T-Statistic", "P-Value"]).T

# Apply the decision to both weekend and weekday
weekend_results_df['Decision'] = weekend_results_df['P-Value'].apply(hypothesis_decision)
weekday_results_df['Decision'] = weekday_results_df['P-Value'].apply(hypothesis_decision)

# display the results
print("Hypothesis Test Results - Weekends:\n", weekend_results_df)
print("\nHypothesis Test Results - Weekdays:\n", weekday_results_df)
