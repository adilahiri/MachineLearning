# code for House Prices: Advanced Regression Techniques - Aditya Lahiri
# import the required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
import numpy as np
from scipy.stats import skew
# Load the test and training data
train_dataset= pd.read_csv('../House_Rent/train.csv')
test_dataset= pd.read_csv('../House_Rent/test.csv')
all_data = pd.concat((train_dataset.loc[:,'MSSubClass':'SaleCondition'],
                      test_dataset.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
#train_dataset["SalePrice"] = np.log1p(train_dataset["SalePrice"])

#log transform skewed numeric features:
#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#
#skewed_feats = train_dataset[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#
#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Convert the categorical data and remove NaN
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
train_features = all_data[:train_dataset.shape[0]]
test_features = all_data[train_dataset.shape[0]:]
train_label = train_dataset.SalePrice
# Feature scaling not required taken care of in Multipleregression class

# Scaling Data
x=train_features
y=train_label
from sklearn.preprocessing import StandardScaler 
sc_X= StandardScaler()
sc_Y= StandardScaler()
train_features_sc= sc_X.fit_transform(x) 
train_label_sc=sc_Y.fit_transform(y)
test_features_sc=sc_X.fit_transform(test_features)
# apply model to test 
regressor = LinearRegression()
regressor.fit(train_features,train_label)
svr_reg= SVR(kernel ='rbf',C=100)
svr_reg.fit(train_features_sc,train_label_sc)
#predict the test result
Y_Pred_Linear= regressor.predict(test_features)
Y_pred_svr = sc_Y.inverse_transform(svr_reg.predict(test_features_sc))
Y_actual = train_label
Y_AVG1= (Y_Pred_Linear + Y_pred_svr)/2
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(Y_AVG1,Y_actual))
#Y_pred=np.expm1(Y_pred)
#np.savetxt('submission.csv', (Y_AVG1))
