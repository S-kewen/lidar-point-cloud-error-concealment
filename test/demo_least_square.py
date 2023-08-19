import numpy as np
from numpy import mat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

if __name__ == "__main__":
    X = np.random.uniform(0,360,100000)
    Y = np.random.uniform(-16.8,10,100000)


    Y_mat = mat(Y).T

    X_temp = np.ones((5, 3))
    X_temp[:, 0] = X[:, 0]
    X_temp[:, 1] = X[:, 1]
    #print(X_temp)
    X_mat = mat(X_temp)
   # print(X_mat)
    pamaters = (((X_mat.T) * X_mat).I) * X_mat.T * Y_mat
    #print(pamaters)


    a = 10000
    b = 20
    s = np.array([[a, b]])
    s_temp = np.ones((1, 3))
    s_temp[:, 0] = s[:, 0]
    s_temp[:, 1] = s[:, 1]
    #print(s_temp)
    s_mat = mat(s_temp)
    m = s_mat * pamaters
    print("m: ", m)
