import gudhi
import numpy as np
import matplotlib.pyplot as plt
import time
time_start = time.time()  # 记录开始时间
plt.rcParams['font.sans-serif'] = ['SimHei']

def draw_complex(complex1, point_cloud):
    points = []
    edges = []
    triangles = []
    for i in range(len(complex1)):
        if len(complex1[i][0]) == 1:
            points.append(complex1[i][0][0])
        elif len(complex1[i][0]) == 2:
            edges.append(complex1[i][0])
        elif len(complex1[i][0]) == 3:
            triangles.append(complex1[i][0])

    plt.figure()
    # triangles
    for i in range(len(triangles)):
        one = triangles[i][0]
        two = triangles[i][1]
        three = triangles[i][2]
        X = [point_cloud[one][0], point_cloud[two][0], point_cloud[three][0], point_cloud[one][0]]
        Y = [point_cloud[one][1], point_cloud[two][1], point_cloud[three][1], point_cloud[one][1]]

        plt.fill(X, Y, color='blue', alpha=0.3)

    # edges
    for i in range(len(edges)):
        one = edges[i][0]
        two = edges[i][1]
        X = [point_cloud[one][0], point_cloud[two][0]]
        Y = [point_cloud[one][1], point_cloud[two][1]]

        # plt.plot(X, Y, color='red', alpha=0.4)

    # points
    X = []
    Y = []
    for i in range(len(points)):
        one = points[i]
        X.append(point_cloud[one][0])
        Y.append(point_cloud[one][1])
    # label = []
    # for index, value in enumerate(point_cloud):
    #     if value[0]==X[i]
    #     label.append(index)
    # file = open('./Y.txt', "r")
    # row = file.readlines()
    # file.close()
    # for i in range(len(X)):
    #     plt.text(X, Y, row[i], fontproperties='SimHei')#标注对应点
    plt.scatter(X, Y, color='green')

    single = []
    link = []
    for edge in edges:
        link.append(edge[0])
        link.append(edge[1])
    for point in points:
        if point not in link:
            single.append(point)
            print(point_cloud[point][0], point_cloud[point][1])
    # print(len(edges))
    # print(len(link))



def draw_barcode(filename):

    point_cloud = np.loadtxt(filename)

    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=8.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    ############改了gudhi.plot_persistence_barcode
    diag = simplex_tree.persistence()
    _, persistence = gudhi.plot_persistence_barcode(diag)

    num = 0
    barcode = []
    for interval in reversed(persistence):
        if interval[0] == 1:
            barcode.append(interval[1])
            num += 1
    barcode.sort(key=lambda x: x[1], reverse=False)
    filter_idx = round(num / 4 * 3)
    filtration = barcode[filter_idx][0]
    filtration = str(filtration).split('.')[0] + '.' + str(filtration).split('.')[1][:1] ##保留一位不进位
    filtration = float(filtration)
    print("filtration", filtration)
    # print(type(filtration))
    # print(barcode)


    rip_complex = []

    for filtered_value in simplex_tree.get_filtration():
        if filtered_value[1] <= filtration:
            # print(filtered_value)
            rip_complex.append(filtered_value)

    # print(rip_complex)

    draw_complex(rip_complex, point_cloud)

    plt.title("持续同调聚类")
    plt.show()


if __name__ == '__main__':
    filename = 'D:\Ablation study\medical-ai-master\词向量\src\dis_vector.txt'
    # filename = './wiki1_200.txt'
    draw_barcode(filename)


    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # print(time_sum)