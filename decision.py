import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('finaldata.csv')
X = df.drop(['date', 'energy_kWh'], axis=1)
y = df['energy_kWh after normalisation']  # Replace with your continuous target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree Regression model
decision_tree = DecisionTreeRegressor(random_state=42)  # You can adjust hyperparameters as needed
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred = decision_tree.predict(X_test)

# Evaluate the model using R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared score on the testing data with Decision Tree Regression:", r2)
