# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:39:24 2021

@author: Zhashenghao
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
 
# vecA,vecB是数组形式
# 欧式距离
def distEula(vecA, vecB):
    return sum((vecA - vecB)**2)**0.5

def randCent(dataSet, k):
    n = np.shape(dataSet)[1] #获取数据集列数
    cluster_center = np.mat(np.zeros((k, n)))#初始化聚类中心矩阵 k*n
    for column in range(n):
        mincol = min(dataSet[:, column])
        maxcol = max(dataSet[:, column])
        #random.rand(k, 1):产生一个0-1之间的随机数向量（k行1列）
        cluster_center[:, column] = np.mat(mincol + float(maxcol - mincol) * np.random.rand(k,1))
    return cluster_center

def KmeansAlg(dataset, k):
    # 样本的个数
    m = np.shape(dataset)[0] 
    # 保存每个样本的聚类情况，第一列表示该样本属于某一类，第二列是与聚类中心的距离
    clusterAssment = np.mat(np.zeros((m,2))) 
    # 产生随机质心,将列表形式转换为数组形式
    centroids = np.array(randCent(dataset, k))
    # 控制聚类算法迭代停止的标志，当聚类不再改变时，就停止迭代
    clusterChanged = True 
    while clusterChanged:  
        # 先进行本次迭代，如果聚类还是改变，最后把该标志改为True，从而继续下一次迭代
        clusterChanged = False 
        for i in range(m): # 遍历每一个样本
            # 每个样本与每个质心计算距离
            # 采用一趟冒泡排序找出最小的距离，并找出对应的类
            # 计算与质心的距离时，刚开始需要比较，记为无穷大
            mindist = np.inf
            for j in range(k): # 遍历每一类
#                 print(np.array(dataset[i,:]))
#                 print(centroids)
                distj = distEula(dataset[i,:],centroids[j,:])
                if distj<mindist:
                    mindist = distj
                    minj = j
            # 遍历完k个类，本次样本已聚类
            if clusterAssment[i,0] != minj:  # 判断本次聚类结果和上一次是否一致
                clusterChanged = True   # 只要有一个聚类结果改变，就重新迭代
            clusterAssment[i,:] = minj,mindist**2  # 类别，与距离
        # 外层循环结束，每一个样本都有了聚类结果
        
        # 更新质心
        for cent in range(k):
            # 找出属于相同一类的样本
            data_cent = dataset[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(data_cent,axis=0)
    return centroids,clusterAssment

if __name__ == "__main__":   
    dataset=[[73,40,7],[60,15,5],[61,19,2],[34,18,6],
             [67,26,10],[91,40,4],[101,40,13],[81,40,6],
             [88,40,8],[122,40,17],[102,50,17],[87,50,12],
             [116,50,11],[110,50,17],[164,50,17],[40,30,1],
             [76,40,17],[118,50,9],[160,50,15],[96,50,16]]
    dataset = np.array(dataset) #转化为矩阵
    print(KmeansAlg(dataset, 3))
    
    
    
    
    
    
    
    
    
