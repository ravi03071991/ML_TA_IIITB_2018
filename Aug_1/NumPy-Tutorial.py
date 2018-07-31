
# coding: utf-8

# #### Introduction to Numpy

# In[1]:

# Load the numpy library
import numpy as np


# In[2]:

# Check the version
np.__version__


# In[3]:

# Create a list of 10 numbers
create_list = list(range(10))


# In[4]:

print("Items in the list: ", create_list)


# In[5]:

# Function to print datatypes of list
def print_dtypes(l):
    for item in l:
        print(type(item))


# In[6]:

# Check the datatypes of the list
print_dtypes(create_list)


# In[7]:

# Let's try to create list with different datatypes
list_hetrogeneous = [1, "2", "A", 3, {"A" :3, "B":4}, 9, (10, 20, 30, 40), True]


# In[8]:

# Print hetrogeneous list
print("Heterogeneous List:\n", list_hetrogeneous)


# In[9]:

# Check the Datatypes in the hetrogeneous list
print_dtypes(list_hetrogeneous)


# #### Creating Numpy arrays

# Unlike lists, numpy arrays are homogeneous in nature that means they comprise of only one data type like integer/ float/ double/ string/ boolean etc...

# In[10]:

# Let's try to create numpy array with different datatypes
np.array([[1, 2, 3],
         ["A", "B", "C"]])


# As you can see above numpy converted integers to strings to make array homogeneous

# Let's try to create some numpy arrays

# In[11]:

# creating array with 5 one's of type int
np.ones(5, dtype='int')


# In[12]:

# creating array with 5 zero's of type float
np.zeros(5, dtype='float')


# In[13]:

#Lets create an identity matrix of 2x2
np.eye(2)


# Understanding the importance of np.random.seed()

# In[14]:

# create a 5x5 array with elements taken from normal distribution of mean 0 and standard deviation 1 
# Setting random.seed()
np.random.seed(10)
x = np.random.normal(0, 1, (5,5))
x


# In[15]:

# create a 5x5 array with elements taken from normal distribution of mean 0 and standard deviation 1 
x = np.random.normal(0, 1, (5,5))
x


# As you can see x contains different values than we created above since we did not set seed value.
# With different seed value numpy generates different numbers.

# In[16]:

# create a 5x5 array with elements taken from normal distribution of mean 0 and standard deviation 1 
np.random.seed(20)
x = np.random.normal(0, 1, (5,5))
x


# Set the same seed value if you want to generate same random numbers. 
# Let's try with seed=10.

# In[17]:

# create a 5x5 array with elements taken from normal distribution of mean 0 and standard deviation 1 
np.random.seed(10)
x = np.random.normal(0, 1, (5,5))
x


# seed = 20

# In[18]:

# create a 5x5 array with elements taken from normal distribution of mean 0 and standard deviation 1 
np.random.seed(20)
x = np.random.normal(0, 1, (5,5))
x


# In[19]:

# Check the shapes of created numpy arrays
print("Shape of x is:", x.shape)


# #### Array Indexing 

# Indexing in python starts from 0.

# In[20]:

# Let's create 1-D array with alphabets
x = np.array(["A", "B", "C", "D", "E", "F", "G"])


# In[21]:

# Get value of first element
x[0]


# In[22]:

# Get value at second element
x[1]


# In[23]:

# Get value of first element from last 
x[-1]


# In[24]:

# Get value of second element from last 
x[-2]


# In[25]:

# Let's create 2-D array
np.random.seed(10)
y = np.random.normal(0, 1, (2,2))
y


# In[26]:

# Get value of 1st row and 1st column
y[0, 0]


# In[27]:

# Get value of last row and last column
y[-1, -1]


# #### Array Slicing 

# Access multiple elements of the array 

# In[28]:

# Create 1-D array of 20 elements from 10-30
x = np.arange(10, 30)
x


# In[29]:

# Access elements till 4th position
x[ : 5]


# In[30]:

# Access elements from 10th position
x[10 : ]


# In[31]:

# Access elements from 14th to 19th position
x[14 : 19]


# In[32]:

# Reverse the array
x[ : : -1]


# #### Array Concatenation

# Array Concatenation = Combining 2 or more arrays, that's it.

# In[33]:

# You can concatenate two or more arrays at once.

arr_1 = np.array([1,2,3])
arr_2 = np.array([4,5,6])
arr_3 = np.array(['A','B', 'C'])

np.concatenate([arr_1, arr_2, arr_3])


# In[34]:

# You can also use this function to create 2-dimensional arrays.

matrix = np.array([["I", "Love", "ML"],["not", "you", "girl"]])
print(matrix)
np.concatenate([matrix, matrix])


# In[35]:

# Using its axis parameter, you can define row-wise or column-wise matrix.

# Example showing concatinating two 2-d arrays one below the other.

concatenated_matrix_axis_0 = np.concatenate([matrix, matrix],axis=0)

print(concatenated_matrix_axis_0)

print (" size of concatenated_matrix_axis_0 = ", concatenated_matrix_axis_0.shape)


# In[36]:

# Example showing concatinating two 2-d arrays one after/adjacent the other

concatenated_matrix_axis_1 = np.concatenate([matrix, matrix],axis=1)

print(concatenated_matrix_axis_1)

print ("size of concatenated_matrix_axis_1 =", concatenated_matrix_axis_1.shape)


# Until now, we used the concatenation function of arrays of equal dimension. 

# How can we combine a 1-D array and 2-D array.

# np.vstack or np.hstack help us in such situations. 

# Let us see how.

# In[37]:

# Stack using vstack

x = np.array([9, 10, 11])
matrix = np.array([[12, 13, 14],[15, 16, 17]])
vstack_example = np.vstack([x, matrix])

print(vstack_example)
print("Shape of stacked array:", vstack_example.shape)


# In[38]:

# Stack using hstack

z = np.array([[0], [0]])
hstack_example = np.hstack([matrix, z])

print(hstack_example)
print("Shape of stacked array:", hstack_example.shape)


# Excercise: 

#     1) Create a 1-D array and split them into three 1-D arrays.
#     2) Create a 2-D array of size 10x10 and split them into two 2-D arrays.

# Hint: Explore np.split() and np.vsplit()

# In[ ]:



