import numpy as np
import pandas as pd
import tensorflow as tf
from ann_visualizer.visualize import ann_viz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=12, activation="relu"))

ann.add(tf.keras.layers.Dense(units=12, activation="relu"))

ann.add(tf.keras.layers.Dense(units=24, activation="relu"))

ann.add(tf.keras.layers.Dense(units=24, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = ann.fit(x_train, y_train, batch_size=32, epochs=50)

ann_viz(ann, title="My first neural network")

tf.keras.utils.plot_model(ann, show_shapes=True, show_layer_names=True)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(accuracy_score(y_test, y_pred))
