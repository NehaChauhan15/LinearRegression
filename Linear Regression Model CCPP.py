#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import dataset
df=pd.read_csv(r"Downloads/combined_cycle_power_plant.csv",sep=';')


# In[3]:


df


# In[4]:


df.head()


# Data Exploration , Domain Knowledge,gather more relevant data from business or from outside 
# 
# Data Cleaning -check for missing values ,check for duplicates ,check for un realistic values 
# 
# Feature Engineering
# Feature Extraction-extract inportant information from data
# Feature Selection- select the most important feature to predict label
# 
# Pre Processing- changing format of data so that it can be processed by Machine Learning algorithm
# 
# Apply Machine Learning Algorithm
# 
# Performance Analysis
# 
# Tuning - improvement in performance of ML model
# 
# Export the trained model for production
# 
# Deployement to Production
# 
# Monitoring of performance in Production
# 
# 

# In[5]:


# Gives statistical information
df.describe()


# In[6]:


df.shape   # To find the no of rows n columns


# In[7]:


df.dtypes  # To find the datatype


# In[8]:


df.isnull().sum()  # To find the null value


# In[9]:


df=df.drop_duplicates()   #To drop the duplicates


# In[10]:


df.shape   # Again check the shape now it varies from above as we have removed duplicate values


# In[11]:


##EDA Part - We are taking conti bcoz my data is in continous form.
##We are converting all to list
conti=df.columns.tolist()   


# In[12]:


conti


# In[14]:


# Here we have to create histogram for each varaible
for i in conti:
    print(df[i].describe())   # Statistical description of each variable. here i means each variable of list
    sns.histplot(x=i,data=df,color='y') # x is representing each variable i and data is the file created
                                       #data is the parameter and df is variable passed
    plt.show()


# In[15]:


# Here we check the behavoiur of independent variable on dependent variable
# Basically we are generating a comparion chart  here
for i in conti:
    sns.scatterplot(x=i,y='energy_output',data=df) # X has independent variable n y has dependent variable
    plt.show()
# First graph tells EO is max at lowest temp so -ve relation is there
# Second graph tells EO is genertaing in clusters ie 35-45, 45-55, 55-80 but overall behavoiur is -ve ie if we increae EO
  # my exhaust vaccum decline
# Third graph tells us if we increase the value of ambient pressure my EO is also increasing & we are getting max o/p at
#  120-130 but in range of 1000-1020 the energy is almost stable means it goes up and down
# Fourth graph tells us +ve relation as when we increase value of relative humidity my EO is also increasing
# Fifth graph tells us when we compare with self it is always a linear graph


# Here above in EDA we have seen that 2 variales are having -ve relation & 2 are having +ve relation. so in order to find the 
# values of -ve and +ve we need to find co realtion of variables

# In[16]:


cor=df.corr()


# In[17]:


cor


# In[18]:


sns.heatmap(cor,annot=True)
plt.show()   
# Darker the color represent -ve corelation n lighter the color represent +ve corelation


# In[19]:


# Implement the Algorithm
# To work on it we need x& y where x has all independent variable while y is dependent variable
x=df.drop('energy_output',axis=1)
y=df['energy_output']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8) 
# Majorly train & test size is of three type
#Train size       Test Size
#75               25
#80               20
#90               10
# We can also mention test size also ie test_size=0.25 


# In[22]:


print(xtrain.shape)
print(ytrain.shape)


# In[23]:


print(xtest.shape)
print(ytest.shape)


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


# Create the object for class
model=LinearRegression()


# In[26]:


# Fit is used to make the algo learn from the train dataset
model.fit(xtrain,ytrain)


# In[27]:


# Now we apply the formulae ie y=mx+c where m is coefficiant & c is intercept
model.intercept_


# In[28]:


# As we know there is 4 values of x so it is m1,m2,m3,m4
model.coef_
 
# Also we get differnet values from other bcoz train & test data is done in random manner ie have different data points


# In[29]:


# Now create one variable ie ypred for predicting y value so pass it xtest
ypred=model.predict(xtest)


#Remenber prediction is done on xtest & accuracy is calculated on ytest


# In[30]:


ypred


# In[31]:


# Assumptions to be made that my model is performing good
#Plot a scatter plot for regression
plt.scatter(ytest,ypred)

# in graph we are getting that datapoints are in linear order so this means that we are going in correct directions


# In[32]:


# calculate Residual means error
residual=ytest-ypred
residual


# In[33]:


# Plot this residual
sns.distplot(residual,kde=True)


# In[34]:


plt.scatter(ypred,residual)


# In[35]:


from sklearn.metrics import mean_absolute_error,r2_score

# Mean absolute error gives error part of the model while R2 score will give accuracy of model


# In[36]:


print(mean_absolute_error(ytest,ypred))


# In[37]:


score =(r2_score(ytest,ypred))
print(score)
#R2 score is the value we get when we apply the linear regression. This value tells us how close the data is to the regression line


# In[38]:


# calculate Adjusted R2 so for that no such lib is there so just apply the formula
1-(1-score)* (len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)


# here adjusted r2 is less than r2 . this means model is good


# Then we apply joblib bcoz we have already completed the dataset and then we want to add new values to the dataset
# so in order to check the accuracy we will use this

# In[40]:


import joblib


# In[41]:


joblib.dump(model,'ccpp-model.pkl')


# In[42]:


test1=[[20,100,90,260]]


# In[43]:


var=joblib.load('ccpp-model.pkl')


# In[44]:


test1_pred=var.predict(test1)


# In[45]:


print(test1_pred)


# In[ ]:




