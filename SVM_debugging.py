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


# In[4]:


x = df.drop('Phase', axis=1)
y=df[["BatchID","Phase"]]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[6]:


BatchIDX_test=X_test[['BatchID']]
X_train2=X_train.drop('BatchID',axis=1)
X_test2=X_test.drop('BatchID',axis=1)

BatchIDy_test=y_test[['BatchID']]
y_train2=y_train.drop('BatchID',axis=1)
y_test2=y_test.drop('BatchID',axis=1)


# In[7]:


X_train2.head(3)


# In[8]:


y_train2.head(2)


# In[9]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train2,y_train2.iloc[:,0])


# In[10]:


y_pred = svclassifier.predict(X_test2)


# In[11]:


print(confusion_matrix(y_test2,y_pred))
print(classification_report(y_test2,y_pred))


# In[16]:


pickle.dump(svclassifier, open('G:\\svm.pkl','wb'))


# In[ ]:




