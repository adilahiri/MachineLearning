#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import training and testing data
train_dataset = pd.read_csv("train.csv")
test_dataset=pd.read_csv("test.csv")
# Seperate the train and test features
Train_Mat= train_dataset.iloc[:,1:].values
Test_Mat= test_dataset.iloc[:,:].values
# Seperate the training labels
Train_Label = train_dataset.iloc[:,0].values
# Check for missing values
imputer = Imputer(missing_values="NaN",strategy ="mean",axis=0)
imputer=imputer.fit(Train_Mat[:,:])
Training_Mat= Train_Mat
Training_Mat[:,:]=imputer.transform(Train_Mat[:,:])
#Standardizing the test and training set
sc_X= StandardScaler()
Training_Mat_Sc= sc_X.fit_transform(Training_Mat)
Test_Mat_Sc=sc_X.transform(Test_Mat)
# Find the principal components
pca=PCA(n_components=50)
Training_Mat_pca=pca.fit_transform(Training_Mat_Sc)
Test_Mat_pca=pca.fit_transform(Test_Mat_Sc)
explained_variance= pca.explained_variance_ratio_ 
#Setting up the classifier model
Classifier = LogisticRegression(multi_class='multinomial',solver= 'lbfgs')
Classifier.fit(Training_Mat_pca,Train_Label)
# predicting test set result
PredictedTestLabel = Classifier.predict(Test_Mat_pca) 
# write to csv
np.savetxt('submissionpca.csv', (PredictedTestLabel))