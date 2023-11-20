import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# read the CSV file, specifying the date column to be parsed as datetime object
df = pd.read_csv('finaldata.csv', parse_dates=['date'])

# check for NaN and infinite values in the data
if df.isnull().values.any():
    df = df.dropna()
if not np.isfinite(df).all().all():
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

# remove any outliers in the data
z = np.abs(stats.zscore(df[['energy_kWh after normalisation',
                            'day_of_month', 'month', 'week', 'weekday', 'season']]))
df = df[(z < 3).all(axis=1)]

# create the observation sequence and normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(df[['energy_kWh after normalisation',
                         'day_of_month', 'month', 'week', 'weekday', 'season']])

# split the data into training and validation sets
X_train, X_val = train_test_split(X, test_size=0.2, random_state=123)

# initialize the HMM model and try different numbers of hidden states and covariance types
best_accuracy = 0
best_model = None
for num_states in range(2, 6):
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        model = hmm.GaussianHMM(n_components=num_states,
                                covariance_type=cov_type, n_iter=10000)
        model.fit(X_train)
        hidden_states = model.predict(X_val)
        accuracy = sum(hidden_states == 0) / len(hidden_states)
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

# print the best model's accuracy on the validation set
if best_model is not None:
    hidden_states = best_model.predict(X_val)
    accuracy = sum(hidden_states == 0) / len(hidden_states)
    print(f"Best model's accuracy on validation set: {accuracy}")
else:
    print("No best model found")
