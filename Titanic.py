#### Titanic Dataset - Aditya Lahiri
#import joblib
#from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#Get the dataset and split it into test and train features

##Import the Libraries 

   
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_features=train.drop(['PassengerId','Ticket','Name','Cabin'], axis=1)
test_features=test.drop(['PassengerId','Ticket','Name','Cabin'], axis=1)


Y_train = (train_features.iloc[:,0].values)
X_train = (train_features.iloc[:,1:].values)
X_test =  (test_features.iloc[:,0:].values)

# deal with  missing data
imputer = Imputer(missing_values="NaN",strategy ="mean",axis=0)
imputer=imputer.fit(X_train[:,2:6])
X_train[:,2:6]=imputer.transform(X_train[:,2:6])
X_test[:,2:6] = imputer.transform(X_test[:,2:6])

## Remove NA from embark features
embark_train = Counter(X_train[:,-1].flat).most_common(1)
embark_test = Counter(X_test[:,-1].flat).most_common(1)
X_train= pd.DataFrame(X_train)
X_test= pd.DataFrame(X_test)
X_train[6]=X_train[6].fillna(value=embark_train[0][0])
X_test[6]=X_test[6].fillna(embark_test[0][0])

# Encoding The categorical data
X_train=X_train.iloc[:,:].values
X_test=X_test.iloc[:,:].values
LE = LabelEncoder()

X_train[:,1]=LE.fit_transform(X_train[:,1]) 
X_test[:,1]=LE.fit_transform(X_test[:,1]) 

X_train[:,6]=LE.fit_transform(X_train[:,6]) 
X_test[:,6]=LE.fit_transform(X_test[:,6]) 


onehotencoder = OneHotEncoder(categorical_features = [1,6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test= onehotencoder.fit_transform(X_test).toarray()




X_train= pd.DataFrame(X_train)
X_test= pd.DataFrame(X_test)

# Get rid of unwanted redundant dummies 
X_train = X_train.drop([0,2], axis=1)
X_test = X_test.drop([0,2], axis=1)

X_train_xgb = X_train
X_test_xgb = X_test
# Scaling the data
#Standardizing the test and training set
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Create the crossvalidation set
#X_traincv, X_testcv, y_traincv, y_testcv = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

# Classigier design 
CLF = []
CLF1=[]
random_state = 2
CLF.append(SVC(C=20,kernel='rbf',random_state=2))
CLF.append(LogisticRegression())
CLF.append(KNeighborsClassifier(n_neighbors=5, metric ='minkowski',p =2))

# Classifier Evaluation
cv_results = []
for CLF in CLF:
    cv_results.append(cross_val_score(CLF, X_train, y = Y_train, scoring = "accuracy", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
CLF1.append(XGBClassifier())
cv_results.append(cross_val_score(CLF, X_train_xgb, y = Y_train, scoring = "accuracy", cv = kfold))
cv_means_xgb = []
cv_std_xgb = []
cv_means_xgb.append(cv_result.mean())
cv_std_xgb.append(cv_result.std())

# PARAMETER TUNING AND BUILDING THE BEST CLASSIFIER
#SVM
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_
#LR
LR = LogisticRegression()
LR_param_grid= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
gsLR = GridSearchCV(LR,param_grid = LR_param_grid, cv=kfold, scoring="accuracy", verbose = 1)
gsLR.fit(X_train,Y_train)
LR_best = gsLR.best_estimator_
#KNN
KNN= KNeighborsClassifier(n_neighbors=5, metric ='minkowski',p =2)
k = np.arange(1,20,2)
parameters = {'n_neighbors': k}
KNN_param_grid= {}
gsKNN = GridSearchCV(KNN,param_grid = parameters, cv=kfold, scoring="accuracy", verbose = 1)
gsKNN.fit(X_train,Y_train)
KNN_best = gsKNN.best_estimator_
#XGB
CLF1_XGB=XGBClassifier()

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',iid=False, cv=10)
gsearch1.fit(X_train_xgb,Y_train)
CLF1_XGB.fit(X_train_xgb,Y_train)
#Prec
PredictedTestLabel = gsearch1.predict(X_test_xgb) 
# write to csv
np.savetxt('submissionXGB1.csv', (PredictedTestLabel))





