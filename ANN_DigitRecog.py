import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
# import training and testing data
train_dataset = pd.read_csv("train.csv")
test_dataset=pd.read_csv("test.csv")
# Seperate the train and test features
Train_Mat= train_dataset.iloc[:,1:].values # 1:
Test_Mat= test_dataset.iloc[:,:].values
# Seperate the training labels
Train_Label = train_dataset.iloc[:,0].values
Train_Label=Train_Label.T
#labelencoder_TC = LabelEncoder()
#Train_Label[:,0] = labelencoder_TC.fit_transform(Train_Label[:,0])
#encoder_y= LabelEncoder.Transform()
#onehotencoder = OneHotEncoder(categorical_features = [0])
#Train_Label_Cat = onehotencoder.fit_transform(Train_Label).toarray()
#Train_Label_Cat = Train_Label_Cat[:, 1:]

dummy_y = np_utils.to_categorical(Train_Label)
# Check for missing values
imputer = Imputer(missing_values="NaN",strategy ="mean",axis=0)
imputer=imputer.fit(Train_Mat[:,:])
Training_Mat= Train_Mat
Training_Mat[:,:]=imputer.transform(Train_Mat[:,:])
#Standardizing the test and training set
sc_X= StandardScaler()
Training_Mat_Sc= sc_X.fit_transform(Training_Mat)
Test_Mat_Sc=sc_X.transform(Test_Mat)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 397, init = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 397, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(Training_Mat_Sc, dummy_y, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(Test_Mat_Sc)
y_pred = (y_pred > 0.5)
Y_Class=np.zeros(shape=(28000,1))
t=np.zeros(shape=(28000,1))
#for i in range (27999):
#   t[i]= np.where(y_pred[i,:])[0][0]
#   Y_Class[i]=t[0][0]
p1=np.column_stack(np.where(y_pred[0:28000,:]))  
ypred2=y_pred[27417:27999+1,:]
p2=np.column_stack(np.where(ypred2))
ypred3=y_pred[27990:,:]
p3=np.column_stack(np.where(ypred3))
p1=p1[:,1]
p2=p2[:,1]
p3=p3[:,1]
Y_Class= np.concatenate((p1,p2,p3))
np.savetxt('submissionartinn.csv', (Y_Class[:,1]))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Test_Mat_Sc, y_pred)