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
            oneInstance = np.reshape(oneInstance,(33,13))
            h1,h2,v1,v2 = 0,33,0,13
            for j in range(oneInstance.shape[0]):
                if(np.all(oneInstance[j]==0)):
                    global h1
                    h1 = j+1
                else:
                    break
            for j in range(oneInstance.shape[0]):
                if(np.all(oneInstance[(oneInstance.shape[0]-j-1)]==0)):
                    global h2
                    h2 = oneInstance.shape[0]-j-1
                else:
                    break
            for j in range(oneInstance.shape[1]):
                if(np.all(oneInstance[:,j]==0)):
                    global v1
                    v1 = j+1
                else:
                    break
            for j in range(oneInstance.shape[1]):
                if(np.all(oneInstance[:,oneInstance.shape[1]-j-1]==0)):
                    global v2
                    v2 = oneInstance.shape[1]-j-1
                else:
                    break
            newInstance = oneInstance[h1:h2,v1:v2]
            h,v = newInstance.shape
            newInstance = transform.resize(newInstance,(32,32))
            newInstance = np.reshape(newInstance,(np.product(newInstance.shape),))
            newInstance = np.append(newInstance,[h,v])
            newInstance = np.insert(newInstance,0,original_data[i][0:2])
            original_data_transformed.append(newInstance)

cosTransform(train_data,train_data_transformed,train_error_data)
cosTransform(test_data,test_data_transformed,test_error_data)

train_data_transformed = np.array(train_data_transformed)
test_data_transformed = np.array(test_data_transformed)

np.savetxt(path+"trainV2.csv",train_data_transformed,fmt="%f",delimiter=",")
np.savetxt(path+"testV2.csv",test_data_transformed,fmt="%f",delimiter=",")
