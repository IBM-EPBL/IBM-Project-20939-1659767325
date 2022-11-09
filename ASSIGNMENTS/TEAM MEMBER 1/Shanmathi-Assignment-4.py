#!/usr/bin/env python
# coding: utf-8

# # 1.Download the dataset 

# In[1]:


#Dataset downloaded


# # 2. Load the Dataset

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv('Mall_Customers.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns


# # Visualizations
# 

# # Univariate Analysis

# In[7]:


sns.displot(df['Age'],color= 'green',bins=20)
plt.title('Age distribution plot')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[8]:


sns.displot(df['Annual Income (k$)'],color= 'orange',bins=20)
plt.title('Annual Income distribution plot')
plt.xlabel('Annual Income')
plt.ylabel('Count')
plt.show()


# In[9]:


sns.displot(df['Spending Score (1-100)'],color= 'purple',bins=20)
plt.title('Spending Score distribution plot')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.show()


# In[10]:


plt.figure(figsize=(20,8))
sns.countplot(x=df['Age'])
plt.title('Age count plot')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[11]:


plt.figure(figsize=(20,8))
sns.countplot(x=df['Annual Income (k$)'])
plt.title('Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Count')
plt.show()


# In[12]:


plt.figure(figsize=(20,8))
sns.countplot(x=df['Spending Score (1-100)'])
plt.title('Spending Score count plot')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.show()


# In[13]:


sns.boxplot(x=df['Age'])
plt.title('Age box plot')
plt.xlabel('Age')
plt.show()


# In[14]:


sns.boxplot(x=df['Annual Income (k$)'])
plt.title('Annual Income box plot')
plt.xlabel('Annual Income')
plt.show()


# In[15]:


sns.boxplot(x=df['Spending Score (1-100)'])
plt.title('Spending Score')
plt.xlabel('Spending Score')
plt.show()


# # Bi-Variate Analysis

# In[16]:


sns.lineplot(x=df['Age'],y=df['Spending Score (1-100)'])


# In[17]:


sns.lineplot(x=df['Age'],y=df['Annual Income (k$)'])


# In[18]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],hue=df['Gender'],
                palette= ['red','green'] ,alpha=0.6)
plt.title('Distribution of Gender based on Annual Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# # Multi-Variate Analysis

# In[19]:


sns.pairplot(df,hue='Gender')


# In[20]:


plt.figure(figsize=(12,8));
sns.heatmap(df.corr(), cmap="YlGnBu",annot=True);


# # 4. Descriptive statistics

# In[21]:


df.describe()


# # 5. Handle the Missing values

# In[22]:


df.isnull().sum()


# # 6. Find the outliers and replace the outliers

# In[23]:


Q1 = df['Annual Income (k$)'].quantile(0.25)
Q3 = df['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR 

Q1,Q3,IQR,lower_bound,upper_bound


# In[24]:


df[(df['Annual Income (k$)'] < lower_bound) | (df['Annual Income (k$)'] > upper_bound)]


# In[25]:


df[(df['Annual Income (k$)'] > lower_bound) & (df['Annual Income (k$)'] < upper_bound)]


# In[26]:


df = df[(df['Annual Income (k$)'] > lower_bound) & (df['Annual Income (k$)'] < upper_bound)]


# In[27]:


sns.boxplot(x=df['Annual Income (k$)'])


# # 7. Check for Categorical columns and perform encoding

# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])


# In[29]:


df.head()


# # 8. Scaling the data

# In[30]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
data_scaled[0:5]


# # 9. Perform any of the clustering algorithms

# In[31]:


Income_Spend = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(Income_Spend)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[32]:


km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(Income_Spend)

plt.figure(figsize=(15,8))
plt.scatter(Income_Spend[y_means == 0, 0], Income_Spend[y_means == 0, 1], s = 100, c = 'red', label = 'Average')
plt.scatter(Income_Spend[y_means == 1, 0], Income_Spend[y_means == 1, 1], s = 100, c = 'yellow', label = 'Spenders')
plt.scatter(Income_Spend[y_means == 2, 0], Income_Spend[y_means == 2, 1], s = 100, c = 'purple', label = 'Best')
plt.scatter(Income_Spend[y_means == 3, 0], Income_Spend[y_means == 3, 1], s = 100, c = 'magenta', label = 'Low Budget')
plt.scatter(Income_Spend[y_means == 4, 0], Income_Spend[y_means == 4, 1], s = 100, c = 'orange', label = 'Saver')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.legend()
plt.title('Customere Segmentation using Annual Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# # 10. Add the cluster data with the primary dataset

# In[33]:


pd.Series(y_means)


# In[34]:


df['Result'] = pd.Series(y_means)


# In[35]:


df.head()


# # 11. Split the data into dependent and independent variables

# In[36]:


df.info()


# In[37]:


x = df.iloc[:,1:5]


# In[38]:


x.head()


# In[39]:


y = df['Result']


# In[40]:


y.head()


# # 12. Split the data into training and testing

# In[41]:


x.shape, y.shape


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train.head(2)


# In[43]:


x_train.shape


# In[44]:


x_test.head(2)


# In[45]:


x_test.shape


# In[46]:


y_train.head(2)


# In[47]:


y_train.shape


# In[48]:


y_test.head(2)


# In[49]:


y_test.shape


# # Build the Model, Train the Model and Test the Model

# In[50]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()

param = {
    'max_depth':[3,6,9,12,15],
    'n_estimators' : [10,50,100,150,200] 
}

rf_search = RandomizedSearchCV(rf,param_distributions=param,n_iter=5,scoring=make_scorer(mean_squared_error),
                               n_jobs=-1,cv=5,verbose=3)

rf_search.fit(x_train, y_train)


# In[51]:


means = rf_search.cv_results_['mean_test_score']
params = rf_search.cv_results_['params']
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))
    if mean == min(means):
        print('Best parameters with the minimum Mean Square Error are:',param)


# In[52]:


rf = RandomForestRegressor(n_estimators=10, max_depth=9)
rf.fit(x_train,y_train)

rf_pred =  rf.predict(x_test)


# # 16. Measure the performance using Metrics

# In[53]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))
print('MSE:', metrics.mean_squared_error(y_test, rf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
print('R2 Score :',metrics.r2_score(y_test,rf_pred))


# In[ ]:




