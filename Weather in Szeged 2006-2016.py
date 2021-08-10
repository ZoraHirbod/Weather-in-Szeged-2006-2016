#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv(r'C:\Users\zhirb005\Downloads\archive (1)\weatherHistory.csv')


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


sns.pairplot(df)


# In[11]:


sns.distplot(df['Temperature (C)'],kde=False)


# In[12]:


sns.jointplot(x='Humidity',y='Temperature (C)',data=df)


# In[15]:


sns.jointplot(x='Humidity',y='Apparent Temperature (C)',data=df)


# In[13]:


df.corr()


# In[14]:


sns.heatmap(df.corr())


# In[16]:


df.columns


# In[20]:


X=df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)','Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover','Pressure (millibars)']]


# In[19]:


y=df['Apparent Temperature (C)']


# In[21]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lm=LinearRegression()


# In[26]:


lm.fit(X_train,y_train)


# In[27]:


lm.coef_


# In[28]:


print(lm.intercept_)


# In[32]:


df1=pd.DataFrame(lm.coef_,X.columns)


# In[33]:


df1


# In[35]:


predictions=lm.predict(X_test)


# In[36]:


plt.scatter(y_test,predictions)


# In[37]:


from sklearn import metrics


# In[46]:


print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('MSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:




