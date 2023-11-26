import numpy as np

arr = np.array([[1, 3], [1, 2]])

x = np.argwhere(arr == [1, 3])
print(arr)
print(x)

array1 = np.array([1, 2, 3])
array1 = np.array(array1).reshape(-1, 1)
array2 = np.array([5, 6, 7])
array2 = np.array(array2).reshape(-1, 1)
array3 = np.array([5, 6, 8])
array3 = np.array(array3).reshape(-1, 1)
# padrão axis=0
matriz = np.concatenate((array1, array2, array3), axis=1)

print(matriz)
