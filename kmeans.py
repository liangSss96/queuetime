import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


dataset = pd.read_csv('aggregation10.csv', index_col=0)
data = dataset.values
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
name = data[:, -1]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
ax.set_zlabel('median')
ax.set_ylabel('std')
ax.set_xlabel('mean')
# for i in range(len(data)):
#     ax.text(x[i], y[i], z[i], name[i])
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[:, 0:3])
dataset['label'] = kmeans.labels_
dataset.to_csv('aggregation10.csv', index=False)
# colors = ['r', 'b', 'g', 'black']
# markers = ['o', 's', 'D', '*']
# fig = plt.figure()
# ax = Axes3D(fig)
# for i, l in enumerate(kmeans.labels_):
#     ax.scatter(x[i], y[i], z[i], color=colors[l], marker=markers[l])
# plt.show()