import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = train.drop(labels=["Survived", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

y_train = train["Survived"]

x_test = test.drop(labels=["Name", "Ticket", "Cabin", "Embarked"], axis=1)


lb = LabelBinarizer()
ss = StandardScaler()

x_train["Sex"] = lb.fit_transform(x_train["Sex"])
x_test["Sex"] = lb.transform(x_test["Sex"])

x_train["Age"].fillna(value=0, inplace=True)
x_test.fillna(value=0, inplace=True)

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

#clf = GridSearchCV(SVC(), parameters)

clf = SVC()

clf.fit(x_train, y_train)
a = clf.predict(x_test)

print(a)

np.savetxt("survivors.txt", a, fmt='%s')





