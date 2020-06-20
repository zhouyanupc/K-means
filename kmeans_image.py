# -*- coding: utf-8 -*-
# 使用K-means对图像进行聚类, 显示分割标识的可视化
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing

# 加载图像,并对数据进行规范化
def load_dat(filepath):
    # 读文件
    f = open(filepath, 'rb')
    data = []
    # 获取图像像素值
    img = image.open(f)
    # 获取图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 获取点(x, y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height

# 加载图像,获得图片尺寸和规范化的img
img, width, height = load_dat('113312.jpg')
# 用K-means进行图像2聚类
kmeans = KMeans(n_clusters=2)
print('开始聚类分割')
kmeans.fit(img)
label = kmeans.predict(img)
print('生成图片分割结果')
# 将图像的聚类结果转化成图像尺寸矩阵
label = label.reshape([width, height])
# 创建新图像,保存聚类结果
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1)) - 1)
pic_mark.save('113312_mark.jpg', 'JPEG')