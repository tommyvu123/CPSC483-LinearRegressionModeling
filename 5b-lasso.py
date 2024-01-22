import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time

#Data to be read
df = pd.read_csv('Data1.csv')

#Divide into dependent and independent
features = ['T', 'P', 'TC', 'SV']
X = df[features].copy()
y = df['Idx']

current_time = time.time()

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

X[features] = min_max_scaler.fit_transform(X)

order = 4
for power in range(2, order + 1):
    for feature in features:
        X[f'{feature}^{power}'] = X[feature] ** power
        X[f'{feature}^{power}'] = X[feature] ** power

# Train model
model = linear_model.LassoCV(alphas = [0.1, 0.01, 0.001], cv = 10)
model.fit(X, y)

training_time = "{:.2f}".format(time.time() - current_time)

#Test model with training data
y_hat = model.predict(X)
RMSE = "{:.5f}".format(mean_squared_error(y, y_hat, squared = False)) 
R_2 = "{:.5f}".format(r2_score(y, y_hat))

print(f'alpha: {model.alpha_}, Training Time: {training_time} secs, RMSE: {RMSE}, R^2: {R_2}')
