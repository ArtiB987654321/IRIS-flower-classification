#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import scipy.stats as st


# In[2]:


iris = pd.read_csv('IRIS.csv')


# In[4]:


iris.head()


# In[5]:


# to display stats about data
iris.describe()


# In[6]:


iris.info()


# In[8]:


# to display no.of samples on each class
iris['species'].value_counts()


# In[9]:


# check for null values
iris.isnull().sum()


# In[10]:


iris['sepal_length'].hist()


# In[11]:


iris['sepal_width'].hist()


# In[12]:


iris['petal_length'].hist()


# In[13]:


iris['petal_width'].hist()


# In[14]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','50Iris-virginica']


# In[16]:


for i in range(3):
    x = iris[iris['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("sepal_width	")
    plt.legend()


# In[17]:


for i in range(3):
    x = iris[iris['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.legend()


# In[18]:


for i in range(3):
    x = iris[iris['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
    plt.xlabel("sepal_length")
    plt.ylabel("petal_length")
    plt.legend()


# In[19]:


for i in range(3):
    x = iris[iris['species'] == species[i]]
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
    plt.xlabel("sepal_width")
    plt.ylabel("petal_width")
    plt.legend()


# In[20]:


iris.corr()


# In[23]:


corr = iris.corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[25]:


iris['species'] = le.fit_transform(iris['species'])
iris.head()


# In[72]:


from sklearn.model_selection import train_test_split
#train - 70
#test - 30
x = iris.drop(columns=['species'])
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


# In[73]:


# Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[74]:


model.fit(x_train, y_train)


# In[75]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[66]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[76]:


model.fit(x_train, y_train)


# In[77]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[69]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[70]:


model.fit(x_train, y_train)


# In[79]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




