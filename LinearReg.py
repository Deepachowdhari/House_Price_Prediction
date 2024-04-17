#!/usr/bin/env python
# coding: utf-8

# In[73]:

import numpy as np
import pandas as pd
import pickle
import warnings


# In[74]:


df=pd.read_csv("House_Price.csv")


# In[46]:


df.head()


# In[47]:


X=df[["bedrooms","bathrooms","sqft_living","condition","yr_built"]]


# In[48]:


X


# In[49]:


y=df.price


# In[50]:


y


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


regressor=LinearRegression()


# In[55]:


regressor.fit(X_train,y_train)


# In[56]:


y_pred=regressor.predict(X_test)


# In[57]:


#y_pred


# In[58]:


from sklearn.metrics import mean_squared_error


# In[59]:


mse=mean_squared_error(y_test,y_pred)


# In[60]:


mse


# In[61]:


rmse = np.sqrt(mse)


# In[62]:


rmse


# In[75]:


pickle.dump('regressor',open('regressor_model.pkl','wb'))


# In[76]:


#model=pickle.load(open('regressor_model.pkl','rb'))







