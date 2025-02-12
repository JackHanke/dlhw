
import numpy as np

X = np.array([
    [1, 1,2],
    [1, 2,1],
    [1, 2,3],
    [1, 4,3],
    [1, 10,3],
])

y = np.array([
    [1],
    [1],
    [-1],
    [1],
    [1],
])

# bias is 1
w = np.array([
    [1],
    [0.9],
    [-1.3]
])

print(np.dot(X,w))


