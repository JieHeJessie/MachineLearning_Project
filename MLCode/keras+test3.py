import numpy as np
path = "project/"
trainCSV = path + "train.csv"
train_csv = np.genfromtxt(trainCSV, delimiter=',')
train_label = train_csv[:,1]

index = np.vstack([np.arange(98),np.zeros(98)])
index = index.T
for i in range(train_label.shape[0]):
    j = train_label[i]
    index[j][1] += 1

low_prob_data = []
for i in range(train_data.shape[0]):
    j = train_label[i]
    if(index[j][1]<100):
        low_prob_data.append(i)

train_csv = np.delete(train_csv,low_prob_data,0)

np.savetxt(path+"trainV5.csv",train_data_transformed,fmt="%f",delimiter=",")