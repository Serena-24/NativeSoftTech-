# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:36:47 2024

@author: Serena
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'Hours': [1.1, 2.5, 3.2, 4.5, 5.1, 6.1, 7.4, 8.2, 9.0, 10.5],
    'Scores': [10, 20, 33, 45, 51, 62, 75, 81, 90, 96]
}
df = pd.DataFrame(data)

# Prepare data for training and testing
X = df[['Hours']]  # Input feature
y = df['Scores']   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Plot the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title("Hours Studied vs. Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()

