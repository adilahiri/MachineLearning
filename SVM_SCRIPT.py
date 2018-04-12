import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
# Applying LDA 
lda=LDA(n_components=9)
Training_Mat_LDA=lda.fit_transform(Training_Mat_Sc,Train_Label)
Test_Mat_LDA=lda.transform(Test_Mat_Sc)
ex=lda.explained_variance_ratio_ 
###Setting up the classifier model

# Training and Test for Evaluation 
#X_train,X_test,Y_train,Y_test = train_test_split(Training_Mat_Sc,Train_Label,test_size=0.20,random_state=0)
## Design the classifier on the training set and test it on the training set left for test
Classifier_svm = SVC(kernel='poly',cache_size=7000)
Classifier_svm.fit(Training_Mat_LDA,Train_Label)
####classifier= KNeighborsClassifier(n_neighbors =5, metric ='minkowski',p =2)
####classifier.fit(Training_Mat_Sc,Train_Label)
##### predicting test set result
####PredictedTrainLabel = classifier.predict(Test_Mat_LDA) 
##### Confusion Matrix
####cm= confusion_matrix(Y_test,PredictedTrainLabel)
####fpr, tpr, thresholds = metrics.roc_curve(Y_test, PredictedTrainLabel, pos_label=10)
#### metrics.auc(fpr, tpr)
###
###
###
#PredictedTestLabel = Classifier_svm.predict(Test_Mat_LDA) 
###### write to csv
#np.savetxt('submissionSVMPOLYLDA1.csv', (PredictedTestLabel))