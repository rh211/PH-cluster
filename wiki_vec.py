# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP
import umap
# from mapper import chart
plt.rcParams['font.sans-serif'] = ['SimHei']
file = open("D:\\Ablation study\\word2vec\\word2vec 100\\all.zh.text.vector", "r",encoding='utf-8')
row = file.readlines()
file.close()
# print(len(row))

###########1:200离群点！！！！！
vector = []
for line in row[1:200]:
    line = list(line.strip().split(' '))
    s = []
    for i in line[1:]:
        s.append(i)
    vector.append(s)
# print(len(list_text))
a = np.array(vector)

labels = []
for line in row[1:200]:
    line = list(line.strip().split(' '))
    labels.append(line[0])
# print(labels)

reducer = UMAP(n_neighbors=10, n_components=3, metric='euclidean',n_epochs=1000, learning_rate=1.0, init='spectral',
               min_dist=0.1, spread=1.0, low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1,
               repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0,a=None, b=None,
               random_state=42, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1,
               verbose=False,
               unique=False,
              )
# Fit and transform the data
X = reducer.fit_transform(a)
print(type(X))
# print(type(y))
# Check the shape of the new data
print('Shape of X_trans: ', X.shape)

#############draw plot
# X = X.T
# X0 = X[0]
# X1 = X[1]

# plt.scatter(X0 ,X1 ,s = 30)
# for i in range(len(X0)):
#         x = X0[i]
#         y = X1[i]
#         plt.text(x ,y ,labels[i] ,fontproperties='SimHei')

# plt.title("维基百科词向量化")
# plt.xlabel("x坐标值")
# plt.ylabel("y坐标值")
# plt.savefig(fname="维基百科.png")
# plt.show()
np.savetxt("wiki_3d.txt", X)
