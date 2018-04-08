import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Load the iris dataset
dataset=pd.read_csv("iris.csv")
all_data=dataset.drop("Id",axis=1)

#Look for missing data
all_data.isnull().sum()

# Only need to convert the dependent variable to numerics from categorical form
LE = LabelEncoder()
all_data=all_data.iloc[:,:].values
all_data[:,4]=LE.fit_transform(all_data[:,4]) 
all_data=pd.DataFrame(all_data)