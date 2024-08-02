import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt

# 加载数据
def get_data():
    """
	:return: 数据集、标签、样本数量、特征数量
	"""
    digits = datasets.load_digits(n_class=10)  # 加载8*8的图片
    data = digits.data  # 图片特征
    print(data.shape)
    label = digits.target  # 图片标签
    n_samples, n_features = data.shape  # 数据集的形状
    print(n_samples, n_features)  # 1797,6top
    # 4
    return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.scatter(x=data[i, 0], y=data[i, 1], s=20, color=plt.cm.Set1(label[i] / 10), clip_on=False, label=label[i])
        # fontdict={'weight': 'bold', 'size': 10})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    #避免重复legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #
    plt.title(title, fontsize=18)
    # 返回值
    return fig


# 主函数，执行t-SNE降维
def showSNE(data, label):
    """
    list data[i]=feature map[i] dtype-numpy
    :return:
    """
    data, label, n_samples, n_features = get_data()  # 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(data)
    print(reslut.shape)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
    # 显示图像
    plt.show()
    # plt.savefig('t-SNE-result')


# 主函数
if __name__ == '__main__':
    showSNE(1, 1)
