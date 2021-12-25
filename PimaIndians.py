#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[50]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[5]:


df.shape


# In[3]:


df.head(5)


# In[4]:


#Checking for null values
df.isnull().values.any()


# In[5]:


df = df.append(pd.Series(), ignore_index=True)


# In[6]:


df.isnull()


# In[7]:


if df.isnull().values.any():
    df = df.dropna()


# In[8]:


df.isnull()


# In[11]:


pd.crosstab(index=df['diabetes'], columns='count')


# In[14]:


import sklearn as sk
from sklearn.model_selection import train_test_split


# In[26]:


df_train, df_test = train_test_split(df, test_size=0.25, shuffle=False)


# In[27]:


df_train


# In[28]:


df_test


# In[39]:


df_test.drop('diabetes', inplace=True)
df_test


# In[40]:


y = df_train["diabetes"]
y


# In[41]:


features = ["num_preg",	"glucose_conc",	"diastolic_bp",	"thickness", "insulin",	"bmi", "diab_pred", "age", "skin"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])


# In[45]:



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
predictions


# In[48]:


output = pd.DataFrame({'PersonId': df_test.index, 'diabetes': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[54]:


compare = df["diabetes"]
compare


# In[52]:


output







