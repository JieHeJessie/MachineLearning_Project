import numpy as np

path = "project/"
trainCSV = path + "trainV4.csv"
train_csv = np.genfromtxt(trainCSV, delimiter=',')
testCSV = path + "testV4.csv"
test_csv = np.genfromtxt(testCSV, delimiter=',')

train_label = train_csv[0:30000,1]
train_data = train_csv[0:30000,2:1026]

X_train = train_data
y_train = train_label

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=20 ,max_features=0.2,n_jobs=-1)

clf.fit(train_data,train_label)

y_test = train_csv[30001:50274,1]
X_test = train_csv[30001:50274,2:1026]

print(clf.score(X_test,y_test))

test_index = test_csv[:,0]
test_data = test_csv[:,2:1026]

predicted = clf.predict(test_data)

result = np.vstack([test_index,predicted])
result = result.T
np.savetxt(path+"RF_result.csv",result,fmt="%d",delimiter=",",header="Id,Character")