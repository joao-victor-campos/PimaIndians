#!/usr/bin/env python
# coding: utf-8

# In[100]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[101]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[102]:


df.shape


# In[103]:


df.head(5)


# In[104]:


#Checking for null values
df.isnull().values.any()


# In[105]:


#Adicionar linha vazia para teste
#df = df.append(pd.Series(), ignore_index=True)


# In[106]:


df.isnull()


# In[107]:


#Remoce linha vazia
#if df.isnull().values.any():
 #   df = df.dropna()


# In[108]:


df.isnull()


# In[109]:


pd.crosstab(index=df['diabetes'], columns='count')


# In[110]:


import sklearn as sk
from sklearn.model_selection import train_test_split
X = df.values
Y = df["diabetes"].values


# In[111]:


df_train, df_test, y_train, y_teste = train_test_split(df, Y, test_size=0.25, random_state = 42, shuffle = True)


# In[112]:


df_train


# In[113]:


df_test


# In[114]:



df_test.drop('diabetes', inplace=True, axis=1)
df_test


# In[ ]:





# In[115]:


features = ["num_preg",	"glucose_conc",	"diastolic_bp",	"thickness", "insulin",	"bmi", "diab_pred", "age", "skin"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])


# In[116]:



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y_train)
predictions = model.predict(X_test)


# In[117]:


output = pd.DataFrame({'PersonId': df_test.index, 'diabetes': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[118]:


from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_teste, predictions)))


# In[ ]:





# In[ ]:





# In[ ]:




