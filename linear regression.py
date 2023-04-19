#!/usr/bin/env python
# coding: utf-8

# # load dataset

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("dataset_houseprice.csv")


# In[3]:


df.head(5)


# # explore dataset

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull()


# In[7]:


df.columns


# In[11]:


df.drop(['date','street','city','statezip','country'],axis=1)


# In[12]:


sns.pairplot(df)
plt.show


# In[13]:


def pp(x,y,z):
    sns.pairplot(df, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

pp('bedrooms', 'bathrooms', 'sqft_living')
pp('sqft_lot', 'floors', 'waterfront')
pp('sqft_above', 'sqft_basement', 'yr_built')


# # heatmap

# In[14]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot = True, cmap ='coolwarm', linewidths=2)


# # data preprocessing

# In[15]:


from sklearn.preprocessing import StandardScaler

scaling=StandardScaler()

scaling.fit_transform(df[["sqft_living","sqft_basement","sqft_above"]])


# # implement classifier

# In[18]:


X=df['sqft_living']
X.head(6)


# In[19]:


y=df['price']
y.head(6)


# In[24]:


X = np.array(df['sqft_living']).reshape(-1, 1)
y = np.array(df['price']).reshape(-1, 1)


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1)


# In[26]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[37]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
 
regr.fit(X_train_sc, y_train)
print(regr.score(X_test_sc, y_test))


# In[36]:


y_pred = regr.predict(X_test_sc)
plt.scatter(X_test_sc, y_test)
plt.plot(X_test_sc, y_pred)
 
plt.show()


# # validate on test data

# In[46]:


from sklearn.metrics import r2_score


# In[47]:


r2_score(y_test, y_pred)

