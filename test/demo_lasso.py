import numpy as np
from sklearn.linear_model import Lasso

X = np.random.rand(10, 2)
y = np.random.rand(10)

lasso = Lasso(alpha=0.1)

lasso.fit(X, y)

print('Intercept:', lasso.intercept_)
print('Coefficients:', lasso.coef_)

x_new = np.array([0.5, 0.6])
y_new = np.array([0.7, 0.8])

z_new = lasso.predict([x_new, y_new])

print('Predicted z:', z_new)