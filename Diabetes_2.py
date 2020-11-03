
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Dataset

data= pd.read_csv('C:/Users/nsidd/OneDrive/Desktop/Datasets/diabetes_data_upload.csv')

# Enocding Categorical Values

data['Gender'] = data['Gender'].replace({'Male':1,'Female':2})
data['Polyuria'] = data['Polyuria'].replace({'Yes':1,'No':2})
data['Polydipsia'] = data['Polydipsia'].replace({'Yes':1,'No':2})
data['sudden weight loss'] = data['sudden weight loss'].replace({'Yes':1,'No':2})
data['weakness'] = data['weakness'].replace({'Yes':1,'No':2})
data['Polyphagia'] = data['Polyphagia'].replace({'Yes':1,'No':2})
data['Genital thrush'] = data['Genital thrush'].replace({'Yes':1,'No':2})
data['visual blurring'] = data['visual blurring'].replace({'Yes':1,'No':2})
data['Itching'] = data['Itching'].replace({'Yes':1,'No':2})
data['Irritability'] = data['Irritability'].replace({'Yes':1,'No':2})
data['delayed healing'] = data['delayed healing'].replace({'Yes':1,'No':2})
data['partial paresis'] = data['partial paresis'].replace({'Yes':1,'No':2})
data['muscle stiffness'] = data['muscle stiffness'].replace({'Yes':1,'No':2})
data['Alopecia'] = data['Alopecia'].replace({'Yes':1,'No':2})
data['Obesity'] = data['Obesity'].replace({'Yes':1,'No':2})
data['class'] = data['class'].replace({'Positive':1,'Negative':2})

#Correlation Plot
from pandas import set_option
names = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class']

#Correlation
set_option('display.width', 100)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print(correlations)

array=data.values

X = array[:,0:16]
Y = array[:,16]

#Splitting the Data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
   X,Y,test_size = 0.30, random_state = 42
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Implementing Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
print(accuracy)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

cm

# Random Forrest

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred1 = classifier.predict(X_test)

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred1)
print(accuracy)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmm = confusion_matrix(Y_test, Y_pred1)
print(cmm)

