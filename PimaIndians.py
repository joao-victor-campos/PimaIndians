#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[6]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[8]:


df.shape


# In[10]:


df.head(5)


# In[14]:


#Checking for null values
df.isnull().values.any()


# In[ ]:




