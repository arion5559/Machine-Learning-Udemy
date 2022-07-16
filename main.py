import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 2:].values

dendogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("Dendogram")
plt.xlabel("Custormers")
plt.ylabel("Euclidean Distance")
plt.show()

hical = AgglomerativeClustering(n_clusters=5, affinity="euclidean")
y_hical = hical.fit_predict(x)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(x[y_hical == 0, 0], x[y_hical == 0, 1], x[y_hical == 0, 2], s=100, c="red", label="Cluster 1")
ax.scatter(x[y_hical == 1, 0], x[y_hical == 1, 1], x[y_hical == 1, 2], s=100, c="blue", label="Cluster 2")
ax.scatter(x[y_hical == 2, 0], x[y_hical == 2, 1], x[y_hical == 2, 2], s=100, c="green", label="Cluster 3")
ax.scatter(x[y_hical == 3, 0], x[y_hical == 3, 1], x[y_hical == 3, 2], s=100, c="cyan", label="Cluster 4")
ax.scatter(x[y_hical == 4, 0], x[y_hical == 4, 1], x[y_hical == 4, 2], s=100, c="magenta", label="Cluster 5")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income(k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.legend()

ax.set_title('3d Scatter plot Clusters of custormers')
plt.show()

hical = AgglomerativeClustering(n_clusters=3, affinity="euclidean")
y_hical = hical.fit_predict(x)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(x[y_hical == 0, 0], x[y_hical == 0, 1], x[y_hical == 0, 2], s=100, c="red", label="Cluster 1")
ax.scatter(x[y_hical == 1, 0], x[y_hical == 1, 1], x[y_hical == 1, 2], s=100, c="blue", label="Cluster 2")
ax.scatter(x[y_hical == 2, 0], x[y_hical == 2, 1], x[y_hical == 2, 2], s=100, c="green", label="Cluster 3")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income(k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.legend()

ax.set_title('3d Scatter plot Clusters of custormers')
plt.show()
