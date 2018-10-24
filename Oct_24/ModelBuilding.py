
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


# Dataset link: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

# In[2]:


#reading the dataset
df=pd.read_csv("train.csv")
df.head()


# In[3]:


df.columns


# In[4]:


cols = df.columns

for i in cols:
    if df[i].isnull().sum() != 0:
        print("Column name is: ", i)
        print(df[i].isnull().sum())


# In[5]:


df.dtypes


# In[6]:


#filling missing values
print(df['Gender'].value_counts())
df['Gender'].fillna('Male', inplace=True)

print(df['Married'].value_counts())
df['Married'].fillna('Yes', inplace=True)

print(df['Dependents'].value_counts())
df['Dependents'].fillna(0, inplace=True)

print(df['Self_Employed'].value_counts())
df['Self_Employed'].fillna('No', inplace=True)

print(df.LoanAmount.describe())
df['LoanAmount'].fillna(df.LoanAmount.mean(), inplace = True)

print(df['Loan_Amount_Term'].value_counts())
df['Loan_Amount_Term'].fillna(512, inplace=True)

print(df['Credit_History'].value_counts())
df['Credit_History'].fillna(1.0, inplace=True)


# In[7]:


# Get categorical columns
cat_cols = []
for i in cols:
    if df[i].dtypes == 'object' and i != 'Loan_ID':
        print(i)
        cat_cols.append(i)

# Do label encoding for categorical columns
le = LabelEncoder()
for i in cat_cols:
    df[i] = le.fit_transform(df[i])


# In[8]:


#split dataset into train and test

train, test = train_test_split(df, test_size=0.3, random_state=0)

x_train=train.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_train=train['Loan_Status']

x_test=test.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_test=test['Loan_Status']


# #### LogisticRegression 

# In[9]:


model = LogisticRegression(random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)


# #### NaiveBayes 

# In[10]:


model = GaussianNB()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# #### SVM 

# In[11]:


model = svm.SVC()
model.fit(x_train, y_train)
model.score(x_test, y_test)


# #### Decission Tree 

# In[12]:


model = tree.DecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)


# #### Bagging Classifier 

# In[13]:


model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)


# #### Random Forest 

# In[14]:


model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# In[15]:


# Get feature importance
for i, j in sorted(zip(x_train.columns, model.feature_importances_)):
    print(i, j)


# #### Adaboost 

# In[16]:


model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# #### Gradient boosting classifier 

# In[17]:


model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# #### XGBoost 

# In[18]:


model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# #### LightGBM 

# In[19]:


train_data=lgb.Dataset(x_train,label=y_train)
#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
y_pred=model.predict(x_test)
for i in range(0,185):
    if y_pred[i]>=0.5: 
        y_pred[i]=1
else: 
    y_pred[i]=0
accuracy_score(y_test, y_pred)


# #### CatBoost 

# In[20]:


model=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)


# #### MaxVoting 

# In[21]:


model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model3 = GaussianNB()
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('NB', model3)])
model.fit(x_train,y_train)
model.score(x_test,y_test)


# #### Weighted Averaging 

# In[22]:


model1 = tree.DecisionTreeClassifier()
model2 = GaussianNB()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1 = model1.predict_proba(x_test)
pred2 = model2.predict_proba(x_test)
pred3 = model3.predict_proba(x_test)

weighted_prediction = (0.2*pred1)+(0.4*pred2)+(0.4*pred3)
labelprediction = np.argmax(weighted_prediction, axis = 1)

accuracy_score(labelprediction, y_test)


# #### Stacking 

# In[23]:


def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        
    model.fit(train, y)
    test_pred=model.predict(test)
    return test_pred.reshape(-1,1),train_pred


# In[24]:


model1 = tree.DecisionTreeClassifier(random_state=1)

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=x_train,test=x_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)


# In[25]:


model2 = LogisticRegression()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)


# In[26]:


df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)


# #### Blending 

# In[27]:


train, test = train_test_split(train, test_size=0.2, random_state=0)

x_train=train.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_train=train['Loan_Status']

x_val=test.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_val=test['Loan_Status']

x_val = x_val.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)

model1 = tree.DecisionTreeClassifier()
model1.fit(x_train, y_train)

val_pred1=model1.predict(x_val)
test_pred1=model1.predict(x_test)

val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = LogisticRegression()
model2.fit(x_train,y_train)

val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)

val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)


# In[28]:


df_val = pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test = pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression(random_state=1)
model.fit(df_val,y_val)
model.score(df_test,y_test)


# References:
# 
#     1. https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
#     2. https://mlwave.com/kaggle-ensembling-guide/
#     3. https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
#     4. https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
#     5. https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/
