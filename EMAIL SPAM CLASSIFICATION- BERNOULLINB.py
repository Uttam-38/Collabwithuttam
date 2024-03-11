#!/usr/bin/env python
# coding: utf-8

# # EMAIL SPAM CLASSIFICATION

# 1.IMPORT THE LIBRARIES:

# In[52]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# 2.IMPORT THE DATASET:

# In[50]:


data = pd.read_csv('spam.csv',encoding='latin-1')
data.head()


# 3.DATA PREPROCESSING:

# In[51]:


data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data = data.rename(columns={"v1":"Status","v2":"Mail"})


# 4.CONVERTING TEXT INTO NUMERIC FORM:

# In[54]:


vector = CountVectorizer()
text = vector.fit_transform(data["Mail"])


# 5.SPLITTING OF DATASET:

# In[55]:


X_train, X_test, y_train, y_test = train_test_split(text, data["Status"], test_size=0.3,random_state=33)


# 6.APPLYING THE MODEL:

# In[81]:


model = BernoulliNB()


# 7.FITTING THE DATA INTO MODEL:

# In[82]:


model.fit(X_train,y_train)


# 8.PREDICITION OF TEST SET:

# In[83]:


pred = model.predict(X_test)
pred


# 9.FINDING THE ACCURACY OF THE MODEL:

# In[84]:


accuracy=model.score(X_test,y_test)
accuracy


# 10.PREDICTION ON NEW DATA:

# In[85]:


new_emails=[
    "Claim the Rupees 1 lakh as a reward for you!!",
    "Confidential about the seminar"
    ]


# In[86]:


new_emails_transformed=vector.transform(new_emails)


# In[87]:


new_predictions= model.predict(new_emails_transformed)


# In[88]:


new_predictions


# Thus, the classification of emails into ham and spam is done by the classifier Bernoulli Naive Bayes model with an accuracy of 97%.
