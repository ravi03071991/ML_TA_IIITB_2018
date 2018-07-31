
# coding: utf-8

# # Pandas and Matplotlib tutorial

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

# Load iris dataset
#filename = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#data = pd.read_csv(filename, sep=',', header=None)
data = pd.read_csv('Iris-2.csv')


# In[3]:

# Basic dataset statistics
data.describe()


# In[4]:

# Info about datatypes and missing values
data.info()


# In[5]:

# Removing null values
data.dropna().head()


# In[6]:

# Get column names of data
cols = list(data.columns)
print(cols)


# In[7]:

# slicing data frames
n = 10
first_n = data[:n]
print(first_n)


# In[8]:

# slicing based on index of columns
# returns nth row 
data.iloc[n-1]


# In[9]:

# returns range of rows and columns
data.iloc[2:10, 1:3]


# In[10]:

# Add a new column 
data['newCol'] = 0
print(data.head())
# Drop the new column
data.drop('newCol', axis='columns', inplace=True)
data.head()


# In[11]:

# Applying a function on the data

sepal_lengths = data['SepalLengthCm'] 
mean = np.mean(sepal_lengths)

# Define function 
do = lambda x : x - mean

mean_sepal = data['SepalLengthCm'].apply(do)
print(mean_sepal.head())


# In[12]:

# sorting data
data.sort_values(by=['PetalLengthCm'], ascending=True, inplace=False).head()


# In[13]:

# Finding the number of data points in each class
data['Species'].value_counts()


# In[14]:

# Cumulative sum of the value counts
data['Species'].value_counts().cumsum()


# In[15]:

# dropping duplicate values
data.drop_duplicates().head()


# In[16]:

# removing some columns / row from dataset
data.drop('PetalLengthCm', axis='columns', inplace=False).head()


# Pivot tables

# In[17]:

# Creating pivot tables with np.sum
data.pivot_table(values='SepalLengthCm', index='Species', aggfunc=np.sum)


# In[18]:

# Create pivot table with np.mean
data.pivot_table(values='SepalLengthCm', index='Species', aggfunc=np.mean)


# # Visualizations

# In[19]:

ax = data[data.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='red', label='Setosa')
data[data.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor', ax = ax)
data[data.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Virginica', ax = ax)

ax.set_title("Scatter plot between sepal length and sepal width")


# # Exercises

# 1. Plot Histograms of each species (target label) versus each column name
# 2. Get unique values of each column
# 3. Filter rows based on column values
# 4. Scatter plot between Petal length and Petal width for each Species
# 5. Make a new column with sum of the other columns
# 6. Save a dataframe as a csv, Excel file

# In[ ]:



