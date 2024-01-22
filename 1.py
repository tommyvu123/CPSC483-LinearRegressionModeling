import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time


#Load data
df = pd.read_csv('Data1.csv')

#Split data into independent and dependent
features = ['T', 'P', 'TC', 'SV']
X = df[features]
y = df['Idx']

#Split data
X_training_data, X_testing_data, y_training_data, y_testing_data = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 100)

for order in range(1, 11):
    
    current_time = time.time()
    
  
    X_poly_training_data =  X_training_data.copy()
    X_poly_testing_data = X_testing_data.copy()
    for power in range(2, order + 1):
        for feature in features:
            X_poly_training_data[f'{feature}^{power}'] = X_poly_training_data[feature]**power
            X_poly_testing_data[f'{feature}^{power}'] = X_poly_testing_data[feature]**power
    
    #Train model
    model = linear_model.LinearRegression(copy_X=True)
    model.fit(X_poly_training_data, y_training_data)
    

    training_time = "{:.2f}".format(time.time() - current_time)


    #Test model with training data
    y_hat = model.predict(X_poly_training_data)
    training_RMSE = "{:.5f}".format(mean_squared_error(y_training_data, y_hat, squared = False)) 
    training_R_2 = "{:.5f}".format(r2_score(y_training_data, y_hat))
    
    #Test model with testing data
    y_hat = model.predict(X_poly_testing_data)
    testing_RMSE = "{:.5f}".format(mean_squared_error(y_testing_data, y_hat, squared = False)) 
    testing_R_2 = "{:.5f}".format(r2_score(y_testing_data, y_hat))

    print(f'Order: {order}, Training Time: {training_time} secs, Testing RMSE: {testing_RMSE}, Training RMSE: {training_RMSE}, Testing R^2: {testing_R_2}, Training R^2: {training_R_2}')
