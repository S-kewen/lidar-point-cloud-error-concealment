from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time

import numpy as np

col1 = np.random.rand(10000)

col2 = np.random.rand(10000)

col3 = np.random.randint(1, 4, size=10000)

data = np.concatenate([col1.reshape(-1, 1), col2.reshape(-1, 1), col3.reshape(-1, 1)], axis=1)


dt = DecisionTreeClassifier(max_depth=5)
svm = SVC(kernel="rbf", C=10)
rf = RandomForestClassifier(n_estimators=10, max_depth=3)



new_data = [[0.1, 0.2]]
dt.fit(data[:, :2], data[:, 2])
dt_pred = dt.predict(new_data)
print(f"DT New data {new_data[0]} is predicted as {dt_pred[0]}")

svm.fit(data[:, :2], data[:, 2])
svm_pred = svm.predict(new_data)
print(f"SVM New data {new_data[0]} is predicted as {svm_pred[0]}")

rf.fit(data[:, :2], data[:, 2])
rf_pred = rf.predict(new_data)
print(f"RF New data {new_data[0]} is predicted as {rf_pred[0]}")
    


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

X = np.random.random((1000, 3))
Y = np.random.randint(low=1, high=3, size=(1000), dtype=np.int64)

X = np.round(X, 2)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)


dt = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
start_time = time.process_time()
print('Accuracy of dt classifier on training set: {:.2f}'
     .format(dt.score(X_train, y_train)))
print('Accuracy of dt classifier on test set: {:.2f}'
     .format(dt.score(X_test, y_test)))
print("TOTAL TIME: {}".format(time.process_time() - start_time))

start_time = time.process_time()
svm = SVC(kernel="rbf", C=10).fit(X_train, y_train)
print('Accuracy of svm classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of svm classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
print("TOTAL TIME: {}".format(time.process_time() - start_time))

rf = RandomForestClassifier(n_estimators = 100, max_depth=10).fit(X_train, y_train)
start_time = time.process_time()
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))
print("TOTAL TIME: {}".format(time.process_time() - start_time))
