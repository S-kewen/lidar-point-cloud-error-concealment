import numpy as np
from scipy.optimize import minimize

data = np.random.rand(100000, 3)

def likelihood(params, data):
    z_min, z_max = params
    z_range = z_max - z_min
    log_likelihood = -data.shape[0] * np.log(z_range)
    for x, y, z in data:
        if z_min <= z <= z_max:
            log_likelihood -= np.log(z_range)
        else:
            log_likelihood = -np.inf
            break
    return -log_likelihood

result = minimize(likelihood, [-1, 120], args=(data,), bounds=((None, None), (None, None)))

z_min, z_max = result.x
print("z_min:", z_min)
print("z_max:", z_max)

x_new, y_new = 0.5, 0.8
z_pred = (x_new + y_new) / 2 + (z_max - z_min) / 2
print("z_pred:", z_pred)
