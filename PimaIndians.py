#!/usr/bin/env python
# coding: utf-8

# In[79]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[80]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[64]:


df.shape


# In[65]:


df.head(5)


# In[66]:


#Checking for null values
df.isnull().values.any()


# In[67]:


#Adicionar linha vazia para teste
#df = df.append(pd.Series(), ignore_index=True)


# In[68]:


df.isnull()


# In[44]:


#Remoce linha vazia
#if df.isnull().values.any():
 #   df = df.dropna()


# In[69]:


df.isnull()


# In[70]:


pd.crosstab(index=df['diabetes'], columns='count')


# In[71]:


import sklearn as sk
from sklearn.model_selection import train_test_split
X = df.values
Y = df["diabetes"].values


# In[73]:


df_train, df_test, y_train, y_teste = train_test_split(df, Y, test_size=0.25, shuffle = False)


# In[49]:


df_train


# In[50]:


df_test


# In[74]:



df_test.drop('diabetes', inplace=True, axis=1)
df_test


# In[ ]:





# In[75]:


features = ["num_preg",	"glucose_conc",	"diastolic_bp",	"thickness", "insulin",	"bmi", "diab_pred", "age", "skin"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])


# In[76]:



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y_train)
predictions = model.predict(X_test)


# In[77]:


output = pd.DataFrame({'PersonId': df_test.index, 'diabetes': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[78]:


from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_teste, predictions)))


# In[ ]:





# In[ ]:





# In[ ]:




