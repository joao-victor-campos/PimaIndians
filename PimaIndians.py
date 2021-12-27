#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


#Checking for null values
df.isnull().values.any()


# In[6]:


#Adicionar linha vazia para teste
#df = df.append(pd.Series(), ignore_index=True)


# In[7]:


df.isnull()


# In[8]:


#Remoce linha vazia
#if df.isnull().values.any():
 #   df = df.dropna()


# In[9]:


df.isnull()


# In[10]:


pd.crosstab(index=df['diabetes'], columns='count')


# In[11]:


import sklearn as sk
from sklearn.model_selection import train_test_split
X = df.values
Y = df["diabetes"].values


# In[25]:


df_train, df_test, y_train, y_teste = train_test_split(df, Y, test_size=0.25, random_state = 42, shuffle = True)


# In[13]:





# In[14]:





# In[32]:



df_test.drop('diabetes', inplace=True, axis=1)
df_train.drop('diabetes', inplace=True, axis=1)


# In[ ]:





# In[16]:


features = ["num_preg",	"glucose_conc",	"diastolic_bp",	"thickness", "insulin",	"bmi", "diab_pred", "age", "skin"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])


# In[36]:


#Primeiro Modelo (RandomForest)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y_train)
predictions = model.predict(X_test)


# In[37]:


output = pd.DataFrame({'PersonId': df_test.index, 'diabetes': predictions})
output.to_csv('submission.csv', index=False)


# In[38]:


from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_teste, predictions)))


# In[39]:


#Segundo modelo (GaussianNB() / naive Bayes)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_gnb = gnb.fit(df_train, y_train).predict(df_test)
print("{0:.4f}".format(metrics.accuracy_score(y_teste, y_pred_gnb)))


# In[30]:





# In[34]:




