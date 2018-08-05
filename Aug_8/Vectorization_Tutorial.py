
# coding: utf-8

# # Vectorization in python

# In[1]:


import numpy as np
import time


# In[2]:


## ndarray - fast and space-efficient array providing vectorized arithmetic operations
data = [6, 7.5, 8, 0, 1]
arr = np.array(data)
print(arr)


# In[3]:


matrix = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr = np.array(matrix)
print(arr)


# In[4]:


# to find the no of dimensions of a numpy matrix
arr.ndim


# In[5]:


# to find the shape of a numpy matrix (returns r,c format)
arr.shape


# In[6]:


# to find the datatype of matrix
arr.dtype


# In[7]:


# initializing vector with zeros
zero_arr = np.zeros(5)
print(zero_arr)

# initializing empty vector
emp_arr = np.empty((2,4))
print(emp_arr)              


# In[8]:


## Regular array operations
arr1 = np.array(np.random.randn(100000))
arr2 = np.array(np.random.randn(100000))
new_arr = np.empty(100000)

# element-wise multiplication
start_time = time.time()
for i in range(100000):
    new_arr[i] = arr1[i]*arr2[i]
end_time = time.time()
print((end_time - start_time)*1000)


# In[9]:


## Array operations without using for loops
# operations between same-sized arrays result in element-wise operations
start = time.time()
mul_arr = arr1 * arr2
end = time.time()
print((end - start)*1000)


# In[10]:


# subtraction between arrays
sub_array = np.empty(100000)
start_time = time.time()
for i in range(100000):
    new_arr[i] = arr1[i]-arr2[i]
end_time = time.time()
print((end_time-start_time)*1000)


# In[11]:


# subtraction using vectorization
start_time = time.time()
sub_array_vec = arr1 - arr2
end_time = time.time()
print((end_time-start_time)*1000)


# In[12]:


# other operations
arr1 = np.array([1,2,3,4])
arr2 = np.array([2,0,1,1])
print(1.0/arr1)
print(arr1 ** 0.5)
print(np.sqrt(arr1))
print(np.exp(arr1))


# In[13]:


## Matrix multiplications
mat1 = np.random.randn(100, 100)
mat2 = np.random.randn(100, 1000)

res = np.empty((100, 1000))
# regular matrix multiplication
start = time.time()
# iterate through rows of X
for i in range(len(mat1)):
   # iterate through columns of Y
   for j in range(len(mat2[0])):
       # iterate through rows of Y
       for k in range(len(mat2)):
            res[i][j] += mat1[i][k] * mat2[k][j]
end = time.time()
print((end-start)*1000)


# In[14]:


# vectorized matrix multiplication
start = time.time()
result = np.matmul(mat1, mat2)
end = time.time()
print((end-start)*1000)


# The time taken by the vectorized operations is drastically reduced compared to the non-vectorized operations
