import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('finaldata.csv')
X = df.drop(['date', 'energy_kWh'], axis=1)
y = df['energy_kWh after normalisation']  # Replace with your continuous target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Polynomial Features
poly_features = PolynomialFeatures(degree=5)  # You can adjust the degree as needed
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create and train the Polynomial Regression model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Make predictions
y_pred = poly_reg.predict(X_test_poly)

# Evaluate the model using R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared score on the testing data with Polynomial Regression of degree 5:", r2)
