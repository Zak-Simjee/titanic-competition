import pandas as pd  # Data Processing
import numpy as np  # Linear Algebra and other functions
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.preprocessing import LabelBinarizer  # Scaler to convert "Sex" to binary
from sklearn.preprocessing import StandardScaler  # Scaler to standardise data


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# Loading train and test datasets

x_train = train.drop(labels=["Survived", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

y_train = train["Survived"]

x_test = test.drop(labels=["Name", "Ticket", "Cabin", "Embarked"], axis=1)
# Setting features to be used in prediction whilst dropping unnecessary features

lb = LabelBinarizer()
ss = StandardScaler()
# Loading scalers

x_train["Sex"] = lb.fit_transform(x_train["Sex"])
x_test["Sex"] = lb.transform(x_test["Sex"])
# Converting "Sex" to binary for both datasets

x_train["Age"].fillna(value=0, inplace=True)
x_test.fillna(value=0, inplace=True)
# Filling missing values in Age with 0 for now. For both datasets

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# Standardising data

clf = SVC()
clf.fit(x_train, y_train)
# Loading and fitting model

np.savetxt("survivors.txt", clf.predict(x_test), fmt='%s')
# Predicting survivors and saving results to "survivors.txt" for submission
