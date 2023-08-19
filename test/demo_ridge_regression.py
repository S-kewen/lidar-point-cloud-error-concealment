from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error


data = np.random.rand(1000, 3)
X_train, X_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2], test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RidgeCV(alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cv=5)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE on test set: ", rmse)

x_new = scaler.transform([[1.0, 2.0]])
z_new = model.predict(x_new)
print("Predicted z for x=1.0, y=2.0: ", z_new[0])


