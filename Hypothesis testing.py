#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv(r'C:\Users\manas\OneDrive\Desktop\taxi data.csv')


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df['Date of Travel']=pd.to_datetime(df['Date of Travel'])
df


# In[9]:


df=df[['Date of Travel','Passenger count','Distance Travelled(KM)','Price Charged','Cost of Trip','Payment_Mode']]


# In[17]:


df


# In[11]:


df['Passenger count'].value_counts(normalize=True)


# In[12]:


df['Payment_Mode'].value_counts(normalize=True)


# In[15]:


# filtering the data based on passenger count between 0 to 6
df=df[(df['Passenger count']>0) & (df['Passenger count']<6)]


# In[20]:


df


# In[22]:


df.describe()


# In[24]:


# checking the outliears
plt.boxplot(df['Price Charged'])


# In[30]:


#finding the upperbond and lowerbond in data. And delting the outliears from data
for col in ['Price Charged','Distance Travelled(KM)']:
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    IQR=q3-q1
    lower_bond=q1-1.5*IQR
    Upper_bond=q1+1.5*IQR
    df=df[(df[col]>=lower_bond) & (df[col]<=Upper_bond)]
    


# In[31]:


df


# In[49]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Distribution of Price Charged')
plt.hist(df[df['Payment_Mode']=='Card']['Price Charged'],histtype='barstacked',bins=20,edgecolor='k',color='#FA643F',label='Card')
plt.hist(df[df['Payment_Mode']=='Cash']['Price Charged'],histtype='barstacked',bins=20,edgecolor='k',color='#FFBCAB',label='Cash')
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
plt.title('Distribution of Distance Travelled(KM)')
plt.hist(df[df['Payment_Mode']=='Card']['Distance Travelled(KM)'],histtype='barstacked',bins=20,edgecolor='k',color='#FA643F',label='Card')
plt.hist(df[df['Payment_Mode']=='Cash']['Distance Travelled(KM)'],histtype='barstacked',bins=20,edgecolor='k',color='#FFBCAB',label='Cash')
plt.legend()
plt.show()


# In[51]:


# Finding Mean and Std of price charge and Distance travelled
df.groupby('Payment_Mode').agg({'Price Charged':['mean','std'],'Distance Travelled(KM)':['mean','std']})


# In[58]:


# calculating the percentage of payment mode usage
plt.title('preference of payment_mode')
plt.pie(df['Payment_Mode'].value_counts(normalize=True),labels=df['Payment_Mode'].value_counts().index,startangle=90,autopct='%1.1f%%')
plt.show()


# In[ ]:


# Null Hypothesis:There is no difference between average price charged between who people are using card and who are using cash
# Alternative Hypothesis: There is a difference between average price charged between who people are using card and who are using cash


# In[66]:


# Checking data follows normal distribution or not
sm.qqplot(df['Price Charged'],line='45')
plt.show()


# In[69]:


Cardsample=df[df['Payment_Mode']=='Card']['Price Charged']
Cashsample=df[df['Payment_Mode']=='Cash']['Price Charged']


# In[71]:


t_stats,p_value=st.ttest_ind(a=Cardsample,b=Cashsample)
print('Ttest_value:',t_stats,'p value:',p_value)


# In[ ]:


#Since the p-value (0.2043) is greater than the significance level (𝛼=0.05α=0.05), we fail to reject the null hypothesis.
#There isn't strong evidence to suggest that the null hypothesis is incorrect based on the given data and the chosen significance level.

