#!/usr/bin/env python
# coding: utf-8

# In[41]:


#importing libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[42]:


#Creating the dataframe
df = pd.read_csv("C:/Users/facla/Documents/CursoPython/Cap11/pima-data.csv")


# In[43]:


df.shape


# In[44]:


df.head(5)


# In[45]:


#Checking for null values
df.isnull().values.any()


# In[46]:


#Adicionar linha vazia para teste
#df = df.append(pd.Series(), ignore_index=True)


# In[47]:


df.isnull()


# In[48]:


#Remoce linha vazia
#if df.isnull().values.any():
 #   df = df.dropna()


# In[67]:


#verificando missing value 
print("# Linhas no dataframe {0}".format(len(df))) 
print("# Linhas missing em glucose_conc {0}".format(len(df.loc[df['glucose_conc'] == 0]))) 
print("# Linhas missing em diastolic_bp {0}".format(len(df.loc[df['diastolic_bp'] == 0]))) 
print("# Linhas missing em thickness {0}".format(len(df.loc[df['thickness'] == 0]))) 
print("# Linhas missing em insulin {0}".format(len(df.loc[df['insulin'] == 0]))) 
print("# Linhas missing em bmi {0}".format(len(df.loc[df['bmi'] == 0]))) 
print("# Linhas missing em age {0}".format(len(df.loc[df['age'] == 0]))) 


# In[50]:


pd.crosstab(index=df['diabetes'], columns='count')


# In[64]:


import sklearn as sk
from sklearn.model_selection import train_test_split
#usando todas as variáveis como preditivas ["num_preg",	"glucose_conc",	"diastolic_bp",	"thickness", "insulin",	"bmi", "diab_pred", "age", "skin"]

# diabetes é variável alvo
Y = df["diabetes"].values


# In[71]:


df_train, df_test, y_train, y_teste = train_test_split(df, Y, test_size=0.25, random_state = 42, shuffle = True)

#Passando as features
features = ["num_preg", "glucose_conc", "diastolic_bp", "thickness", "insulin",	"bmi", "diab_pred", "age"]
X_train = df_train[features]
X_test = df_test[features]
X_test.head(5)


# In[73]:


#Using Impute to substitute missing values 
from sklearn.impute import SimpleImputer

impute_0_to_mean = SimpleImputer(missing_values = 0, strategy = "mean")

X_train = pd.DataFrame(impute_0_to_mean.fit_transform(X_train))
X_test = pd.DataFrame(impute_0_to_mean.fit_transform(X_test))
type(X_test)
X_test.head(100)


# In[59]:





# In[72]:


#Primeiro Modelo (RandomForest)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_teste, predictions)))


# In[77]:


#output the prediction to a csv
output = pd.DataFrame({'PersonId': X_test.index, 'diabetes': predictions})
output.to_csv('submission.csv', index=False)


# In[74]:


#Segundo modelo (GaussianNB() / naive Bayes)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
print("{0:.4f}".format(metrics.accuracy_score(y_teste, y_pred_gnb)))


# In[75]:


#Terceiro modelo (Regressão Logística)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 0.7, random_state = 42)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)
print("{0:.4f}".format(metrics.accuracy_score(y_teste, lr_prediction)))


# In[ ]:





# In[ ]:




