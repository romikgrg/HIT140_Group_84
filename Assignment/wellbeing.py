import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')


# Calculate Descriptive statistics for well-being variables from dataset3(mean, std, min, max, etc.)
wellbeing_stats = dataset3[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr',
                            'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']].describe()

# Display the results

print("\nWell-being Statistics:\n", wellbeing_stats)

# Create histograms for well-being variables
for col in wellbeing_stats:
    plt.figure(figsize=(8, 5))
    plt.hist(dataset3[col], bins=5, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()