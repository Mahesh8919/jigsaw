#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None


# In[4]:


corona=pd.read_csv("C:\\Users\\Ganesh\\Desktop\\CORONA\\patients_data.csv")


# In[27]:


corona.shape


# In[7]:


cor=pd.read_csv("C:\\Users\\Ganesh\\Desktop\\CORONA\\complete.csv")


# In[39]:


corona.head(5)


# In[29]:


corona.isna().sum()


# In[15]:


cor.corr()


# In[22]:


pip install seaborn


# In[23]:


import seaborn as sns


# In[24]:


sns.heatmap(cor.corr())


# In[30]:


### considering only below columns for analysis, deleting rest columns
covid=corona[['p_id','date_announced','age_bracket','detected_state','status_change_date','current_status']].dropna().set_index('p_id')


# In[42]:


covid.dtypes


# In[33]:


len(covid['detected_state'].unique())


# In[45]:


covid=covid[pd.to_numeric(covid['age_bracket'], errors='coerce').notnull()]


# In[46]:


covid.head(5)


# In[47]:


covid.dtypes


# In[49]:


covid['age_bracket']=pd.to_numeric(covid['age_bracket'])


# In[50]:


covid.dtypes


# In[52]:


covid['date_announced']= pd.to_datetime(covid['date_announced'],format='%d/%m/%Y')
covid['status_change_date']= pd.to_datetime(covid['status_change_date'],format='%d/%m/%Y')


# In[73]:


covid.current_status.unique()


# In[56]:


########preprossing the data and encoding string to int#################3
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
state=covid['detected_state'].unique()
le=preprocessing.LabelEncoder()
le.fit(state)
le.classes_
covid['detected_state']=le.transform(covid['detected_state'])


# In[57]:


Ydata=covid[covid['current_status'].isin(['Recovered','Deceased'])]


# In[60]:


Ydata.head(1)


# In[61]:


Ydata=Ydata.drop(['status_change_date', 'date_announced'],axis=1)


# In[62]:


Ydata.head(1)


# In[64]:


X=Ydata.drop(['current_status'],1)
label=Ydata['current_status'].unique()
le.fit(label)
le.classes_
Ydata['current_status']=le.transform(Ydata['current_status'])
y=Ydata['current_status']
print(Ydata.info())


# In[66]:


Ydata.head(1)


# In[68]:


Ydata.current_status.unique


# In[99]:


predict_data=covid[covid['current_status'].isin(['Hospitalized'])]


# In[100]:


predict_data['Days']=(pd.to_datetime('now')-predict_data['date_announced']).dt.days
predict_data=predict_data.drop(['status_change_date', 'date_announced','current_status'],axis=1)


# In[102]:



predict_data=predict_data.drop(['Days'],axis=1)


# In[103]:


predict_data.head(1)


# In[74]:


predict_data.current_status.unique()


# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# In[77]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model=DecisionTreeClassifier(criterion='gini')
model.fit(x_train,y_train)
predict_train=model.predict(x_train)


# In[84]:


predict_train=model.predict(x_test)


# In[79]:


accuracy_test=accuracy_score(y_test, predict_train)


# In[80]:


accuracy_test


# In[104]:


output=model.predict(predict_data) 


# In[105]:


############### decoding i.e. converting int to string("recovered deceased")
output=list(le.inverse_transform(output))


# In[107]:


le.fit(state)
predict_data['detected_state']=list(le.inverse_transform(predict_data['detected_state']))
predict_data['prediction']=output
print("prediction Output: ",predict_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




