#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
os.getcwd()
import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


data = df.drop_duplicates(subset="Species")
data


# In[9]:


df.value_counts("Species")


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


sns.countplot(x="Species",data=df)
plt.show()


# In[12]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm',
                hue='Species', data=df,)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[ ]:


#From the above plot, we can infer that – 

#Species Setosa has smaller sepal lengths but larger sepal widths.
#Versicolor Species lies in the middle of the other two species in terms of sepal length and width
#Species Virginica has larger sepal lengths but smaller sepal widths.


# In[13]:


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm',
                hue='Species', data=df,)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[ ]:


#From the above plot, we can infer that – 

#1. Species Setosa has smaller petal lengths and widths.
#2. Versicolor Species lies in the middle of the other two species in terms of petal length and width
#3. Species Virginica has the largest of petal lengths and widths.


# In[29]:


with pd.option_context('mode.use_inf_as_na', True):
 sns.pairplot(df.drop(['Id'], axis = 1), 
             hue='Species', height=2)



# In[ ]:


#From the above plot, we can infer that – 
#Species Setosa has the smallest of petals widths and lengths. It also has the smallest sepal length but larger sepal widths. 


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['SepalLengthCm'], bins=7)
 
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['SepalWidthCm'], bins=5)
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['PetalLengthCm'], bins=6)
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['PetalWidthCm'], bins=6)


# In[ ]:


#From the above plot, we can see that – 

#The highest frequency of the sepal length is between 30 and 35 which is between 5.5 and 6
#The highest frequency of the sepal Width is around 70 which is between 3.0 and 3.5
#The highest frequency of the petal length is around 50 which is between 1 and 2
#The highest frequency of the petal width is between 40 and 50 which is between 0.0 and 0.5


# In[36]:


plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalLengthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalWidthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalLengthCm").add_legend()
 
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalWidthCm").add_legend()
 
plt.show()


# In[ ]:


#From the above plots, we can see that – 

#In the case of Sepal Length, there is a huge amount of overlapping.
#In the case of Sepal Width also, there is a huge amount of overlapping.
#In the case of Petal Length, there is a very little amount of overlapping.
#In the case of Petal Width also, there is a very little amount of overlapping.


# In[48]:


data1 = df.drop(['Species'], axis = 1)
data1.corr(method='pearson')


# In[50]:


sns.heatmap(data1.corr(method='pearson').drop(['Id'], axis=1).drop(['Id'], axis=0),annot=True)
plt.show()


# In[ ]:


#From the above graph, we can see that –

#Petal width and petal length have high correlations. 
#Petal length and sepal width have good correlations.
#Petal Width and Sepal length have good correlations.


# In[37]:


def graph(y):
  sns.boxplot(x="Species",y=y,data=df)
plt.figure(figsize=(10,10))

plt.subplot(221)
graph('SepalLengthCm')
 
plt.subplot(222)
graph('SepalWidthCm')
 
plt.subplot(223)
graph('PetalLengthCm')
 
plt.subplot(224)
graph('PetalWidthCm')
 
plt.show()


# In[ ]:


#From the above graph, we can see that – 

#Species Setosa has the smallest features and less distributed with some outliers.
#Species Versicolor has the average features.
#Species Virginica has the highest features


# In[42]:


sns.boxplot(x="SepalWidthCm",data=df)


# In[44]:


import sklearn
import pandas as pd
import seaborn as sns


df = pd.read_csv('Iris.csv')

# IQR
Q1 = np.percentile(df['SepalWidthCm'], 25,
				interpolation = 'midpoint')

Q3 = np.percentile(df['SepalWidthCm'], 75,
				interpolation = 'midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df.shape)

# Upper bound
upper = np.where(df['SepalWidthCm'] >= (Q3+1.5*IQR))

# Lower bound
lower = np.where(df['SepalWidthCm'] <= (Q1-1.5*IQR))

# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)

print("New Shape: ", df.shape)

sns.boxplot(x='SepalWidthCm', data=df)

