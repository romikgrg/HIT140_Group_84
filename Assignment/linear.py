import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge datasets on the 'ID' column, using inner join to only keep common respondents
merged_df = pd.merge(df1, df2, on='ID', how='inner')
merged_df = pd.merge(merged_df, df3, on='ID', how='inner')

# Drop rows with missing values
cleaned_df = merged_df.dropna()

# Create new columns for total weekly screen time for computers, video games, smartphones, and TV
cleaned_df['computer'] = cleaned_df['C_wk'] + cleaned_df['C_we']
cleaned_df['videogames'] = cleaned_df['G_wk'] + cleaned_df['G_we']
cleaned_df['smartphone'] = cleaned_df['S_wk'] + cleaned_df['S_we']
cleaned_df['tv'] = cleaned_df['T_wk'] + cleaned_df['T_we']

# Selecting the predictor variables
X = cleaned_df[['computer', 'videogames', 'smartphone', 'tv']]

# Target variable created by calculating the average of well-being indicators
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
y = cleaned_df[wellbeing_columns].mean(axis=1)  # This creates an overall well-being score by averaging

# Check the first few rows of the feature and target data
print(X.head())
print(y.head())


# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing data
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Print the model coefficients to understand how each screen time variable affects well-being
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error and R-squared value
mse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Scatter plot of predicted vs actual well-being scores
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Predicted vs Actual Well-being Scores')
plt.xlabel('Actual Well-being Scores')
plt.ylabel('Predicted Well-being Scores')
plt.show()
