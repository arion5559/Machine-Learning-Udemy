import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = SVC(kernel="linear")
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(accuracy_score(y_test, y_pred))

print(classifier.predict(sc.transform([[1000849, 4, 3, 3, 5, 8, 5, 2, 2, 4]])))
