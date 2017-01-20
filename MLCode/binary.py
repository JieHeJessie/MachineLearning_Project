import numpy as np

path = "project/"
trainCSV = path + "train.csv"
testCSV = path +"test.csv"
train_data = np.genfromtxt(trainCSV,delimiter=',')
test_data = np.genfromtxt(testCSV,delimiter=',')

for i in range(train_data.shape[0]):
    for j in range(9,train_data.shape[1]):
        if train_data[i][j] >127:
            train_data[i][j] =1
        else : 
            train_data[i][j]=0

for i in range(test_data.shape[0]):
    for j in range(9,test_data.shape[1]):
        if test_data[i][j] >127:
            test_data[i][j] =1
        else : 
            test_data[i][j]=0

np.savetxt(path+"trainV1.csv",train_data,fmt="%d",delimiter=",")
np.savetxt(path+"testV1.csv",train_data,fmt="%d",delimiter=",")

