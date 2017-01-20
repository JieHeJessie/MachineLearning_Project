import numpy as np
path = "project/"
train_data = np.genfromtxt(path+"train.csv", delimiter=',')
test_data = np.genfromtxt(path+"test.csv", delimiter=',')

train_data_transformed = []
test_data_transformed = []
train_error_data = []
test_error_data = []

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

def cosTransform(original_data,original_data_transformed,error_data):
    for i in range(original_data.shape[0]):
        oneInstance = original_data[i][9:438]
        oneInstance = oneInstance/255
        if(np.all(oneInstance ==0))or(np.all(oneInstance ==1)):
            error_data.append(original_data[i][0])
        else:
            newInstance = np.reshape(oneInstance,(33,13))
            H = np.sum(newInstance,axis = 1)
            V = np.sum(newInstance,axis = 0)
            newInstance = np.reshape(newInstance,(np.product(newInstance.shape),))
            newInstance = np.append(newInstance,H)
            newInstance = np.append(newInstance,V)
            newInstance = np.insert(newInstance,0,original_data[i][0:2])
            original_data_transformed.append(newInstance)

cosTransform(train_data,train_data_transformed,train_error_data)
cosTransform(test_data,test_data_transformed,test_error_data)

train_data_transformed = np.array(train_data_transformed)

test_data_transformed = np.array(test_data_transformed)

np.savetxt(path+"trainV3.csv",train_data_transformed,fmt="%f",delimiter=",")
np.savetxt(path+"testV3.csv",test_data_transformed,fmt="%f",delimiter=",")