
# coding: utf-8

# ### Data Pre-processing Tutorial

# We will be using House Prices: Advanced Regression Techniques dataset from kaggle.
# Please download "train.csv" file from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data .
# You need to accept the rules of the competition to download the dataset.

# In[1]:

# Import necessary packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


# In[2]:

# Load dataset
data = pd.read_csv("train.csv")


# In[3]:

# Let's see the first 10 rows of dataset
data.head(10)


# In[4]:

# Let's see the last 10 rows of dataset
data.tail(10)


# As you can see there are lot of missing values (NaN) in the dataset

# Let's explore Alley column in the dataset.

# In[5]:

# Check the number of missing values
data.Alley.isnull().sum()


# In[6]:

# Check the frequency of distinct values
data.Alley.value_counts()


# In[7]:

# Let's fill the missing values with Grvl
data.Alley.fillna(value = "Grvl", inplace = True)


# In[8]:

# Check again the missing values
data.Alley.isnull().sum()


# Now let's explore LotFrontage column.

# In[9]:

# Check the number of missing values
data.LotFrontage.isnull().sum()


# In[10]:

# Get the mean of the LotFrontage 
LotFrontage_mean = data.LotFrontage.mean()


# In[11]:

# Fill missing values with mean of the column
data.LotFrontage.fillna(value = LotFrontage_mean, inplace = True)


# In[12]:

# Check again the missing values
data.LotFrontage.isnull().sum()


# Adding a new column with values from existing column

# In[13]:

# Add new column by substracting values of 1st fllor area and 2nd floor area
data["new_column"] = data["1stFlrSF"] - data["2ndFlrSF"]


# In[14]:

# Check the column values
data.new_column


# Did you see something weird?

# In[15]:

# Replace negative values with 0
data.new_column[data.new_column < 0] = 0


# In[16]:

data.new_column


# In[17]:

# Unique values in LotShape column
data.LotShape.unique()


# One-hot and label encoding

# In[18]:

# Encoding Categorical data
labelencoder = LabelEncoder()
target_label = labelencoder.fit_transform(data.LotShape) 
target_label[0:30]


# In[19]:

# One hot encoding of LotShape

pd.get_dummies(data.LotShape,prefix=['LotShape'])


# Splitting dataset into train and test

# In[20]:

# Split dataset into training and testing datasets
train, test = train_test_split(data, test_size = 0.4)


# In[21]:

# Check the original dataset shape
data.shape


# In[22]:

# Check the training dataset shape
train.shape


# In[23]:

# Check the test dataset shape
test.shape


# ### Shuffling the dataset 

# In[24]:

# Shuffle the rows of dataset
data_shuffle = shuffle(data)


# In[25]:

# Check first 5 rows of original dataset
data.head(5)


# In[26]:

# Check first 5 rows of shuffled dataset
data_shuffle.head(5)


# ### Normalizing the data 

# In[27]:

# Normalize SalePrice data
normalizing = StandardScaler()
data["SalePrice_normalized"] = normalizing.fit_transform(data.SalePrice.reshape(-1, 1))


# In[28]:

data.head()


# ### Excercises

#     1) Try filling the missing values of "FireplaceQu".
#     2) Try filling the missing values of "GarageYrBlt" with mean.
#     3) For continuous variables is it good always to fill with mean value of column??
#     4) Filter the dataframe with only continuous variable columns.
#     5) Split test dataset into testing and validation dataset.
#     6) We shuffled the dataset row wise. Try shuffling the dataset column wise.
#     7) Try One-hot encoding on YrSold, SaleType and SaleCondition columns
