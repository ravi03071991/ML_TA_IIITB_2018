
# coding: utf-8

# In[1]:


# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Let us first generate data which follows y = 5x + 6 + delta.

# In[2]:


# Generate data
x = np.random.rand(100)
delta = np.random.rand(100)
y = 5 * x + 6 + delta

print(x.shape)
print(y.shape)

# Plot data
plt.figure(figsize = (10, 5))
plt.plot(x,y, 'b.')
plt.xlabel("Values of x")
plt.ylabel("Values of y")
plt.title("X vs Y")
plt.show()



# Define loss function

# In[3]:


# Loss function
def loss(y, y_predicted):
    return np.mean(np.square(y-y_predicted))


# Define Gradient Descent

# In[4]:


def get_grad(x_data, y_data, m, c):
    m_grad = np.mean((y_data - (m*x_data+c))*(-x_data))
    c_grad = np.mean((y_data - (m*x_data+c))*(-1))
    return m_grad, c_grad


# In[5]:


# Get indices of samples for training, validation and testing
indices = np.random.permutation(x.shape[0])
training_idx, val_idx, test_idx = indices[:60], indices[20:80], indices[80:100]


# In[6]:


# Split x into training and testing data
x_train = x[training_idx]
x_val = x[val_idx]
x_test = x[test_idx]

# Split y into training and testing data
y_train = y[training_idx]
y_val = y[val_idx]
y_test = y[test_idx]


# In[7]:


# Set random.seed value
np.random.seed(40)

# Initialize m and c values
m0 = float(np.random.rand(1))
c0 = float(np.random.rand(1))

print(m0)
print(c0)

# Create empty lists to store intermediate m, c, loss functions
m_vec = []; c_vec = []; loss_val_vec = []; loss_train_vec = []

# NumberOfIterations
NumberOfIterations = 40

# Learning rate
lr = 0.1
for i in range(0, 500):
    m_grad, c_grad = get_grad(x_train, y_train, m0, c0)
    m0 = m0 - lr * m_grad
    c0 = c0 - lr * c_grad
    loss_train = loss(y_train, (m0*x_train+c0))
    loss_val = loss(y_val, (m0*x_val+c0))
    
    m_vec.append(m0)
    c_vec.append(c0)
    loss_val_vec.append(loss_val)
    loss_train_vec.append(loss_train)
    
print("Final m and c values are m_final = {:2.2f}, c_final - {:2.2f}".format(m0,c0))


# In[8]:


# Visualizing change in m and c with number of iterations
plt.figure(figsize=(10,5))
plt.plot(m_vec, label='m')
plt.plot(c_vec, label='c')
plt.title('Change of m and c with iterations')
plt.xlabel("Number of iterations")
plt.ylabel("m/c values")
plt.legend(); plt.show()


# In[9]:


# Visualizing loss with number of iterations
plt.figure(figsize=(10,5))
plt.plot(loss_val_vec, "r.", label = "Validation Loss")
plt.plot(loss_train_vec, "b.", label = "Training Loss")
plt.title('Change in loss with iterations')
plt.xlabel("Number of iterations")
plt.ylabel("loss")
plt.legend(); plt.show()


# In[10]:


# Visualizaing final line on data
m = m0; c = c0
y_pred = m * x_test + c
#y_pred = m * x + c
plt.figure(figsize=(10,5))
plt.plot(x_test, y_test, "b.")
plt.plot(x_test, y_pred, '-', color = 'r')
plt.title('Final values of m & c are m = {:2.2f}, c = {:2.2f}, loss: {:2.2f}'.format(m, c, loss(y_test,y_pred)))
plt.xlabel("Values of x")
plt.ylabel("Values of y")
plt.show()



# It's a long process!!! isn't it?

# Let's use sklearn to build the same regression line 

# In[11]:


# Import necessary packages
import sklearn
from sklearn.linear_model import LinearRegression


# Get Linear Regression model object

# In[12]:


lm = LinearRegression()


# In[13]:


# Fit linear regression on x and y
lm.fit(x_train, y_train)


# It seems dimensions are not correct. Let's check them.

# In[ ]:


# Let us check the shape of x
print(x_train.shape)

# Let us check the shape of xy
print(y_train.shape)


# we need to give shape of (M, N) and not (M, ). So let's reshape them.

# You need to give column vector for x and y

# In[ ]:


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


# In[ ]:


# Shape of x
x.shape


# In[ ]:


# Shape of y
y.shape


# Fit linear regression model on data

# In[ ]:


lm.fit(x, y)


# Get slope and intercept

# In[ ]:


# lm.intercept_
c_pred = lm.intercept_
c_pred


# In[ ]:


# lm.coef_
m_pred = lm.coef_
m_pred


# In[ ]:


# Visualizaing final line on data

y_pred = lm.predict(x_test.reshape(-1,1))
plt.figure(figsize=(10,5))
plt.plot(x_test, y_test, "b.")
plt.plot(x_test, y_pred, "-", color = 'r')
plt.title('Final value of m, c and loss are m = {:2.2f}, c = {:2.2f}, loss: {:2.2f}'.format(m, c, loss(y_test,y_pred)))
plt.xlabel("Values of x")
plt.ylabel("Values of y")
plt.show()



# Using sklearn you need not define loss function and gradient descent. Sklearn takes care of everything.

# ### Excercises 

#     1) Build linear regression from scratch using x1, x2, x3 (3 features) and target label(y).
#     2) Experiement with different learning rates and see how you reach global minima. Does learning rate have effect          on number of iteractions?
#     3) Build linear regression on Kaggle House Prices: Advanced Regression Techniques dataset using any two features          of dataset and sklearn. ( Remember pd.read_csv to load csv file?)
