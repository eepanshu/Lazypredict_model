import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the California housing dataset
data = fetch_california_housing()
print(data.DESCR ) # Print the description of the dataset
print(data.feature_names)  # Print the feature names
print(data.target)  # Print the target variable (median value of homes)
print(data.data)  # Print the feature data
print(data.data.shape)  # Print the shape of the feature data

# Convert the dataset into a pandas DataFrame
california_df = pd.DataFrame(data.data, columns=data.feature_names)
california_df['MEDV'] = data.target  # Add the target variable (median value of homes)

# Select one feature (let's choose 'MedInc' - median income)
X = california_df[['MedInc']]
Y = california_df['MEDV']

print(X.shape)  # Print the shape of X
print(X.head())  # Print the first 5 rows of X
# Split the data into training and testing sets
print(Y.head())  # Print the shape of Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Visualize the results
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)
plt.xlabel('Median Income')
plt.ylabel('Median value of homes (in $100,000s)')
plt.title('Simple Linear Regression on California Housing Dataset')
plt.show()
