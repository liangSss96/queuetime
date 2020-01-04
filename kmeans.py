import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('aggregation10.csv', index_col=0)
print(data)
data = data.values
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
