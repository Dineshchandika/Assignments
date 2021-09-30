#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
st.title("CLAIMANTS")
Gender=st.sidebar.selectbox("Gender",("Female","Male"))
Insurance=st.sidebar.selectbox("Insurance",("claimed","not claimed"))
Seatbelt=st.sidebar.selectbox("Seat belt",("ON","OFF"))


# In[38]:


df=pd.read_csv("C:\\Users\\DELL\\Downloads\\claimants.csv")
df.head()


# In[39]:


df=df.drop(['CASENUM'],axis=1)


# In[40]:


df.head()


# In[41]:


df.shape


# In[42]:


df.isnull().sum()


# In[43]:


df=df.dropna()


# In[44]:


df.shape


# In[45]:


x=df.iloc[:,1:]
y=df.iloc[:,0]


# In[46]:


classifier = LogisticRegression().fit(x,y)
classifier


# In[47]:


y_pred=classifier.predict(x)
y_pred


# In[48]:


y_pred_df=pd.DataFrame({'actual': y,'Predcted':y_pred})
y_pred_df


# In[49]:


pd.crosstab(y,y_pred)


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


accuracy=accuracy_score(y,y_pred)
accuracy


# In[ ]:





# In[ ]:




