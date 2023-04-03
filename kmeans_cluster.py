import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
plt.rcParams['font.sans-serif'] = ['SimHei']

with open('./wiki1_200.txt','r',encoding='utf-8') as f1:
    lines1 = f1.readlines()
length1 = len(lines1)

data1 = np.zeros((length1,2))
for i in range(0, length1):
        vector = lines1[i].split('\n')[0]
        vector1 = vector.split(' ')[0]
        vector2 = vector.split(' ')[1]
        data1[i,0] = vector1
        data1[i,1] = vector2

km1 = KMeans(n_clusters=5, random_state=9)
pred1 = km1.fit_predict(data1)
print(km1.cluster_centers_)#####数据中心

# plt.scatter(data[:, 0], data[:, 1], c=y_pred)
# plt.scatter(data[y_pred == 0, 0], data[y_pred == 0, 1], c='lightgreen', marker='s', label='cluster 1')
# plt.scatter(data[y_pred == 1, 0], data[y_pred == 1, 1], c='orange', marker='o', label='cluster 2')
# plt.scatter(data[y_pred == 2, 0], data[y_pred == 2, 1], c='lightblue', marker='v', label='cluster 3')
# plt.scatter(data[y_pred == 3, 0], data[y_pred == 3, 1], c='blue', marker='d', label='cluster 1')
# plt.scatter(data[y_pred == 4, 0], data[y_pred == 4, 1], c='yellow', marker='p', label='cluster 2')
#
# plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1], c='red', marker='*', label='centroids')
#
# plt.show()
# print(metrics.calinski_harabasz_score(data, y_pred))

######去除离群点前后的数据中心
# [4.01832316 6.94371725]
#  [7.6969315  8.8222638 ]
#  [7.03192425 5.70247221]
#  [9.66522047 9.49651339]
#  [5.68010127 9.87991229]
#
#  [4.01832316 6.94371725]
#  [9.5918364  9.52793018]
#  [7.03192425 5.70247221]
#  [5.64078114 9.8251979 ]
#  [7.67301518 8.76929844]

# plt.boxplot(data[y_pred == 1, 0])
# plt.show()

##################clean
with open('./wiki1_200_clean.txt','r',encoding='utf-8') as f2:
    lines2 = f2.readlines()
length2 = len(lines2)

data2 = np.zeros((length2,2))
for i in range(0, length2):
        vector = lines2[i].split('\n')[0]
        vector1 = vector.split(' ')[0]
        vector2 = vector.split(' ')[1]
        data2[i,0] = vector1
        data2[i,1] = vector2

km2 = KMeans(n_clusters=5, random_state=9)
pred2 = km2.fit_predict(data2)
print(km2.cluster_centers_)#####数据中心

X0 = [data1[pred1 == 0, 0], data2[pred2 == 0, 0]]
X1 = [data1[pred1 == 1, 0], data2[pred2 == 1, 0]]
X2 = [data1[pred1 == 2, 0], data2[pred2 == 2, 0]]
X3 = [data1[pred1 == 3, 0], data2[pred2 == 3, 0]]
X4 = [data1[pred1 == 4, 0], data2[pred2 == 4, 0]]

# 箱型图名称
labels = ["原始数据", "清洗后"]
# 三个箱型图的颜色 RGB （均为0~1的数据）
colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.)]
# 绘制箱型图
# patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = plt.boxplot(X0, patch_artist=True, labels=labels, positions=(1, 1.4), widths=0.3, notch=True)
# 将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

bplot2 = plt.boxplot(X1, patch_artist=True, labels=labels, positions=(2.5, 2.9), widths=0.3, notch=True)
for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

bplot3 = plt.boxplot(X2, patch_artist=True, labels=labels, positions=(4, 4.4), widths=0.3, notch=True)
for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

bplot4 = plt.boxplot(X3, patch_artist=True, labels=labels, positions=(5.5, 5.9), widths=0.3, notch=True)
for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

bplot5 = plt.boxplot(X4, patch_artist=True, labels=labels, positions=(7, 7.4), widths=0.3, notch=True)
for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)



x_position = np.arange(1, 8, 1.5)
print(x_position)
x_position_fmt = ["0", "1", "2", "3", "4"]
plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

plt.xlabel('类别')
plt.ylabel('x坐标值')
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
plt.legend(bplot['boxes'], labels, loc='lower right')  # 绘制表示框，右下角绘制
plt.title("清洗前后x轴坐标点分布")

plt.savefig(fname="x坐标.png")
plt.show()