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

# calculate the overall average
from openpyxl.compat import file


def Average(fileName):
    fi = open(fileName, 'r')
    result = 0.0
    cnt = 0
    for line in fi:
        cnt += 1
        arr = line.split()
        result += int(arr[2].strip())
    return result / cnt


def InerProduct(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result


def PredictScore(av, bu, bi, pu, qi):
    pScore = av + bu + bi + InerProduct(pu, qi)
    if pScore < 1:
        pScore = 1
    elif pScore > 5:
        pScore = 5

    return pScore


def SVD(configureFile, testDataFile, trainDataFile, modelSaveFile):
    # get the configure
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

    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    # qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
    # pu = [[(0.1 * random.random() / temp)  for j in range(factorNum)] for i in range(userNum)]

    eta = [[(numpy.random.laplace(0.0, 20)) for j in range(itemNum)] for i in range(userNum)]

    qi = [[(0.1 / temp) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[(0.1 / temp) for j in range(factorNum)] for i in range(userNum)]

    print("qi %f %f %f %f\n" % (qi[1][1], qi[1][2], qi[2][1], qi[2][2]))
    print("pu %f %f %f %f\n" % (pu[1][1], pu[1][2], pu[2][1], pu[2][2]))
    print("eta %f %f %f %f\n" % (eta[1][1], eta[1][2], eta[2][1], eta[2][2]))

    r1 = 0.1 * random.random() / temp
    # qi = [[r1 for j in range(factorNum)] for i in range(itemNum)]
    r2 = 0.1 * random.random() / temp
    # pu = [[r2 for j in range(factorNum)] for i in range(userNum)]
    # qi = [[r1 for j in range(factorNum)] for i in range(itemNum)]
    # pu = [[r2 for j in range(factorNum)] for i in range(userNum)]
    # print("initialization end\nstart training %f %f\n" %(r1, r2))
    print("initialization end\nstart training \n")

    # train model
    preRmse = 1000000.0
    for step in range(100):
        fi = open(trainDataFile, 'r')
        for line in fi:
            arr = line.split()
            uid = int(arr[0].strip()) - 1
            iid = int(arr[1].strip()) - 1
            score = int(arr[2].strip())
            # print("test_RMSE in step %d %d\n" %(uid, iid))
            prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
            # print("test_RMSE in step1\n")
            eui = score - prediction

            # update parameters
            bu[uid] += learnRate * (eui - regularization * bu[uid])
            bi[iid] += learnRate * (eui - regularization * bi[iid])
            for k in range(factorNum):
                temp = pu[uid][k]  # attention here, must save the value of pu before updating
                pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
                qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])
        fi.close()
        # learnRate *= 0.9
        curRmse = Validate(testDataFile, averageScore, bu, bi, pu, qi)
        print("test_RMSE in step %d: %f" % (step, curRmse))
        if curRmse >= preRmse:
            break
        else:
            preRmse = curRmse

    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    qi = [[(0.1 / temp) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[(0.1 / temp) for j in range(factorNum)] for i in range(userNum)]
    print("qi %f %f %f %f\n" % (qi[1][1], qi[1][2], qi[2][1], qi[2][2]))
    print("pu %f %f %f %f\n" % (pu[1][1], pu[1][2], pu[2][1], pu[2][2]))
    print("start training with Laplace noises\n")

    # train model
    preRmse = 1000000.0

    for step in range(100):
        fi = open(trainDataFile, 'r')
        for line in fi:
            arr = line.split()
            uid = int(arr[0].strip()) - 1
            iid = int(arr[1].strip()) - 1
            score = int(arr[2].strip())
            # print("test_RMSE in step\n")
            prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])

            eui = score - prediction

            # update parameters
            bu[uid] += learnRate * (eui - regularization * bu[uid])
            bi[iid] += learnRate * (eui - regularization * bi[iid])
            for k in range(factorNum):
                temp = pu[uid][k]  # attention here, must save the value of pu before updating
                pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k] - eta[uid][iid] * qi[iid][k])
                qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k] - eta[uid][iid] * temp)
        fi.close()
        # learnRate *= 0.9
        curRmse = Validate(testDataFile, averageScore, bu, bi, pu, qi)
        print("test_RMSE in step %d: %f" % (step, curRmse))
        if curRmse >= preRmse:
            break
        else:
            preRmse = curRmse

    print("qi %f %f %f %f\n" % (qi[1][1], qi[1][2], qi[2][1], qi[2][2]))
    print("pu %f %f %f %f\n" % (pu[1][1], pu[1][2], pu[2][1], pu[2][2]))

    # write the model to files
    fo = file(modelSaveFile, 'wb')
    pk.dump(bu, fo, True)
    pk.dump(bi, fo, True)
    pk.dump(qi, fo, True)
    pk.dump(pu, fo, True)
    fo.close()
    print("model generation over %f %f\n" % (r1, r2))


# validate the model
def Validate(testDataFile, av, bu, bi, pu, qi):
    cnt = 0
    rmse = 0.0
    fi = open(testDataFile, 'r')
    for line in fi:
        cnt += 1
        arr = line.split()
        uid = int(arr[0].strip()) - 1
        iid = int(arr[1].strip()) - 1
        pScore = PredictScore(av, bu[uid], bi[iid], pu[uid], qi[iid])

        tScore = int(arr[2].strip())
        rmse += (tScore - pScore) * (tScore - pScore)
    fi.close()
    return math.sqrt(rmse / cnt)


# use the model to make predict
def Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile):
    # get parameter
    fi = open(configureFile, 'r')
    line = fi.readline()
    arr = line.split()
    averageScore = float(arr[0].strip())
    fi.close()

    # get model
    fi = file(modelSaveFile, 'rb')
    bu = pk.load(fi)
    bi = pk.load(fi)
    qi = pk.load(fi)
    pu = pk.load(fi)
    fi.close()

    # predict
    fi = open(testDataFile, 'r')
    fo = open(resultSaveFile, 'w')
    for line in fi:
        arr = line.split()
        uid = int(arr[0].strip()) - 1
        iid = int(arr[1].strip()) - 1
        pScore = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
        fo.write("%f\n" % pScore)
    fi.close()
    fo.close()
    print("predict over")


if __name__ == '__main__':
    configureFile = 'svd.conf'
    # trainDataFile = 'training0'
    # testDataFile = 'test0'
    trainDataFile = 'u1.test'
    testDataFile = 'u1.test'
    modelSaveFile = 'svd_model.pkl'
    resultSaveFile = 'prediction'

    # print("%f" %Average("ua.base"))
    import time

    start = time.clock()
    SVD(configureFile, testDataFile, trainDataFile, modelSaveFile)
    end = time.clock()
    print(end - start)
# Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile)







