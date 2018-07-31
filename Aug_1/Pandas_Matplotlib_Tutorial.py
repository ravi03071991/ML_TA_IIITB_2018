
# coding: utf-8

# # Pandas and Matplotlib tutorial

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Let's start with Pandas Series

# In[2]:

# Creating a series
a = np.array(np.random.randint(1,100,10))
s = pd.Series(a)
print(s)
print(s.values)


# In[3]:

# Giving Custom Indexes
s = pd.Series(a,index =np.random.randint(400,500,10))
print(s)


# In[4]:

# Creating a Series from a dictionary
dictionary = {'h':123,'g':392,'c':980}
print(pd.Series(dictionary))


# In[5]:

# Accessing elemnets in a Series
# Using indexes and using values

pop_dict = {'Uttar Pradesh': 38332521,'Karnataka': 26448193,'Haryana': 19651127}
pop = pd.Series(pop_dict)
print(pop['Karnataka'])
print(pop[1])


# In[6]:

# Retrieving a range

s = pd.Series(range(10), index = [x for x in 'abcdefghij'])

# Retrieve the first 3 elements

print(s[:3])
print(s[:'c'])

# Retrieve the last element
print(s[-1:])

# Accessing everything but the last element
print(s[:-1])


# Let's work with Iris Dataset

# In[7]:

# Load iris dataset
# filename = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# data = pd.read_csv(filename, sep=',', header=None)
data = pd.read_csv('Iris.csv')


# In[8]:

# Basic dataset statistics
data.describe()


# In[9]:

# Info about datatypes and missing values
data.info()


# In[10]:

# Removing null values
data.dropna().head()


# In[11]:

# Get column names of data
cols = list(data.columns)
print(cols)


# In[12]:

# slicing data frames
n = 10
first_n = data[:n]
print(first_n)


# In[13]:

# slicing based on index of columns
# returns nth row 
data.iloc[n-1]


# In[14]:

# returns range of rows and columns
data.iloc[2:10, 1:3]


# In[15]:

# Add a new column 
data['newCol'] = 0
print(data.head())
# Drop the new column
data.drop('newCol', axis='columns', inplace=True)
data.head()


# In[16]:

# Applying a function on the data

sepal_lengths = data['SepalLengthCm'] 
mean = np.mean(sepal_lengths)

# Define function 
do = lambda x : x - mean

mean_sepal = data['SepalLengthCm'].apply(do)
print(mean_sepal.head())


# In[17]:

# sorting data
data.sort_values(by=['PetalLengthCm'], ascending=True, inplace=False).head()


# In[18]:

# Finding the number of data points in each class
data['Species'].value_counts()


# In[19]:

# Cumulative sum of the value counts
data['Species'].value_counts().cumsum()


# In[20]:

# dropping duplicate values
data.drop_duplicates().head()


# In[21]:

# removing some columns / row from dataset
data.drop('PetalLengthCm', axis='columns', inplace=False).head()


# Pivot tables

# In[22]:

# Creating pivot tables with np.sum
data.pivot_table(values='SepalLengthCm', index='Species', aggfunc=np.sum)


# In[23]:

# Create pivot table with np.mean
data.pivot_table(values='SepalLengthCm', index='Species', aggfunc=np.mean)


# # Visualizations

# In[24]:

ax = data[data.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='red', label='Setosa')
data[data.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor', ax = ax)
data[data.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Virginica', ax = ax)

ax.set_title("Scatter plot between sepal length and sepal width")
plt.show()


# # Exercises

# 1. Plot Histograms of each species (target label) versus each column name
# 2. Get unique values of each column
# 3. Filter rows based on column values
# 4. Scatter plot between Petal length and Petal width for each Species
# 5. Make a new column with sum of the other columns
# 6. Save a dataframe as a csv, Excel file

# In[ ]:



