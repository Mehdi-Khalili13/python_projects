# %%
import numpy as np

# %%
a = np.array([1,2,3,4,5]) # create 1D array
# %%
b = np.array ([[1,2,3], [1,3,6]])
b
# %%
ze = np.zeros((2,3))
ze
# %%
ones = np.ones((2,3))
ones
# %%
rand = np.random.rand (2,3)
rand
# %%
aa = np.arange (0,10,1)
aa
# %%
ln = np.linspace(0,1,10)
ln
# %%
x = np.array([[1.2, 3.4, 5.6],
              [7.8, 9.0, 1.1]])

print(x.shape)   # shape of array
print(x.ndim)    # number of dimensions
print(x.dtype)   # data type

# %%
a = np.array ([1,2,3,4,5,6,7,8,9])
print(a[0])
# %%
print(a[1:3])
print(a[:2])
print(a[1:])
print(a[::3])
# %%
