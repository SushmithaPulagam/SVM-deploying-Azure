#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

   
    
#DATA INPUT- DO cannot be lower than 0% and higher than 100%
data = pd.read_excel ("G:\\Trial_mine\\alldata.xlsx")
df2= data[['BatchID','f_timeh','DO_live','rpm_live','Phase']]
df3=df2.loc[(df2.Phase<3)]#only Phase 1 and 2
df=df3.loc[(df3.rpm_live<=10000) & (df3.DO_live<=100) & (df3.DO_live>=0)] #taking out outliers
#df = df4[df4.index % 50 == 0] #taking every 50 sample (for better diagrams visibility)


# In[2]:


df2.head(5)


# In[3]:


df.head(5)


# In[17]:


final_df = df.drop('BatchID', axis=1)


# In[18]:


X = final_df.drop('Phase', axis=1)
Y =final_df.iloc[:,-1]


# In[19]:


X.head(3)


# In[20]:


Y.head(2)


# In[21]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X,Y)


# In[22]:


pickle.dump(svclassifier, open('G:\\svm1.pkl','wb'))


# In[ ]:




