import numpy as np

x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

a = np.array([2, 10])
a = a.reshape((len(a), 1))
print(a)

print(np.multiply(x, a))
