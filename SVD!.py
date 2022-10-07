# _*_coding : utf-8 _*_
# @Time : 2022/10/4 15:38
# @Author : SunShine
# @File : SVD!
# @Project : SVD
# _*_coding : utf-8 _*_
# @Time : 2022/8/5 9:45
# @Author : SunShine
# @File : SVD
# @Project : SVD

# Ver1.0
# Zero @2014.5.2
#

import math
import random
import numpy
import pickle as pk
import matplotlib.pyplot as plt

# calculate the overall average
from openpyxl.compat import file


def Average(fileName):  # 计算平均值
    fi = open(fileName, 'r')
    result = 0.0
    cnt = 0
    for line in fi:
        cnt += 1
        arr = line.split()
        result += int(arr[2].strip())
    # print(cnt)
    # with open('svd.conf', 'w', encoding='utf-8') as fp:
    #     fp.write(str(cnt)+" "+str(result / cnt))
    return result / cnt


def InerProduct(v1, v2):  # 向量内积
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result


def PredictScore(av, bu, bi, pu, qi):  # 评分
    # 预测评分计算式，
    # av：平均值
    # bu: 用户评分与用户平均的偏差
    # bi: 项目评分与项目平均的偏差
    # pu: 用户特征矩阵
    # qi: 项目特征矩阵
    pScore = av + bu + bi + InerProduct(pu, qi)
    if pScore < 1:
        pScore = 1
    elif pScore > 5:
        pScore = 5

    return pScore


def SVD(configureFile, testDataFile, trainDataFile, modelSaveFile):
    # 从congigure文件中得到用户数、项目数、特征维度、学习率以及正则参数
    fi = open(configureFile, 'r')
    line = fi.readline()
    arr = line.split()
    averageScore = float(arr[0].strip())
    userNum = int(arr[1].strip())
    itemNum = int(arr[2].strip())
    factorNum = int(arr[3].strip())
    learnRate = float(arr[4].strip())
    regularization = float(arr[5].strip())
    fi.close()

    # 初始化模型
    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(userNum)]
    r1 = 0.1 * random.random() / temp
    r2 = 0.1 * random.random() / temp
    print("initialization end\nstart training\n")

    # 训练模型
    s = []
    rmse = []
    preRmse = 1000000.0
    iteration = 500  # 最大训练次数
    for step in range(iteration):
        fi = open(trainDataFile, 'r')
        for line in fi:
            arr = line.split()
            uid = int(arr[0].strip()) - 1
            iid = int(arr[1].strip()) - 1
            score = int(arr[2].strip())
            prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
            # print(pu[uid])
            # print(qi[iid])
            eui = score - prediction  # 误差
            # print(eui)
            # 更新参数
            bu[uid] += learnRate * (eui - regularization * bu[uid])  # 第u个用户的偏离程度
            bi[iid] += learnRate * (eui - regularization * bi[iid])  # 第i个电影的偏离程度
            for k in range(factorNum):
                temp = pu[uid][k]  # attention here, must save the value of pu before updating
                pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
                qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])
        fi.close()
        # learnRate *= 0.9
        curRmse = Validate(testDataFile, averageScore, bu, bi, pu, qi)  # 代入验证模型得到均方根误差
        print("test_RMSE in step %d: %f" % (step, curRmse))
        if curRmse >= preRmse:
            break
        else:
            preRmse = curRmse
        s.append(step)
        rmse.append(curRmse)
    # print(s)   # 次数
    # print(rmse)  # 均方根误差
    # plt.plot(s, rmse)
    # plt.show()
    # return s, rmse

    # write the model to files
    fo = open(modelSaveFile, 'wb')
    pk.dump(bu, fo, True)
    pk.dump(bi, fo, True)
    pk.dump(qi, fo, True)
    pk.dump(pu, fo, True)
    fo.close()
    print("model generation over %f %f\n" % (r1, r2))


# validate the model 验证模型
def Validate(testDataFile, av, bu, bi, pu, qi):
    cnt = 0
    rmse = 0.0
    fi = open(testDataFile, 'r')
    for line in fi:
        cnt += 1
        arr = line.split()
        uid = int(arr[0].strip()) - 1
        iid = int(arr[1].strip()) - 1
        pScore = PredictScore(av, bu[uid], bi[iid], pu[uid], qi[iid])  # 预测的分数
        tScore = int(arr[2].strip())  # 实际的分数
        rmse += (tScore - pScore) * (tScore - pScore)
    fi.close()
    return math.sqrt(rmse / cnt)  # 均方根误差


# use the model to make predict
def Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile):
    # get parameter
    fi = open(configureFile, 'r')
    line = fi.readline()
    arr = line.split()
    averageScore = float(arr[0].strip())
    fi.close()

    # get model
    fi = open(modelSaveFile, 'rb')
    bu = pk.load(fi)
    bi = pk.load(fi)
    qi = pk.load(fi)
    pu = pk.load(fi)
    fi.close()

    # predict
    count = 0
    s = []
    true_score = []
    predict_score = []
    false = []
    fi = open(testDataFile, 'r')
    fo = open(resultSaveFile, 'w')
    for line in fi:
        arr = line.split()
        uid = int(arr[0].strip()) - 1
        iid = int(arr[1].strip()) - 1
        score = int(arr[2].strip())  # 真实分数
        true_score.append(score)
        pScore = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
        predict_score.append(pScore)
        s.append(count)
        count += 1
        false.append(score - pScore)
        fo.write("%f    %f\n" % (pScore, score - pScore))
    fi.close()
    fo.close()

    # 设置图像大小
    fig = plt.figure(figsize=(20, 28), dpi=80)
    # 设置字体
    plt.rcParams['font.sans-serif'] = 'DengXian'
    plt.plot(s[:250], true_score[:250], label="实际")
    plt.plot(s[:250], predict_score[:250], label="预测")
    # plt.plot(s, false)
    print()
    plt.legend()
    plt.show()

    print("predict over")


if __name__ == '__main__':
    # 训练
    # configureFile = 'svd.conf'
    # trainDataFile = 'u1.base'
    # testDataFile = 'u1.test'
    # modelSaveFile = 'svd_model.pkl'
    # resultSaveFile = 'prediction'

    # print("%f" % Average("u1.base"))
    import time

    # start = time.clock()
    # SVD(configureFile, testDataFile, trainDataFile, modelSaveFile)
    # end = time.clock()
    # print("用时 %f 秒" % (end - start))

    # 预测
    configureFile = 'svd1.conf'
    # trainDataFile = 'training0'
    # testDataFile = 'test0'
    testDataFile = 'u4.test'
    modelSaveFile = 'svd_model.pkl'
    resultSaveFile = 'prediction'
    Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile)
