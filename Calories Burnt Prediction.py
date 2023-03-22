
# Importing the Dependencies

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# Data Collection & Processing

# In[ ]:


# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('/content/calories.csv')


# In[ ]:


# print the first 5 rows of the dataframe
calories.head()


# In[ ]:


exercise_data = pd.read_csv('/content/exercise.csv')


# In[ ]:


exercise_data.head()


# Combining the two Dataframes

# In[ ]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)


# In[ ]:


calories_data.head()


# In[ ]:


# checking the number of rows and columns
calories_data.shape


# In[ ]:


# getting some informations about the data
calories_data.info()


# In[ ]:


# checking for missing values
calories_data.isnull().sum()


# Data Analysis

# In[ ]:


# get some statistical measures about the data
calories_data.describe()


# Data Visualization

# In[ ]:


sns.set()


# In[ ]:


# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])


# In[ ]:


# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])


# In[ ]:


# finding the distribution of "Height" column
sns.distplot(calories_data['Height'])


# In[ ]:


# finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])


# Finding the Correlation in the dataset

# 1. Positive Correlation
# 2. Negative Correlation

# In[ ]:


correlation = calories_data.corr()


# In[ ]:


# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# Converting the text data to numerical values

# In[ ]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[ ]:


calories_data.head()


# Separating features and Target

# In[ ]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']


# In[ ]:


print(X)


# In[ ]:


print(Y)


# Splitting the data into training data and Test data

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# XGBoost Regressor

# In[ ]:


# loading the model
model = XGBRegressor()


# In[ ]:


# training the model with X_train
model.fit(X_train, Y_train)


# Evaluation

# Prediction on Test Data

# In[ ]:


test_data_prediction = model.predict(X_test)


# In[ ]:


print(test_data_prediction)


# Mean Absolute Error

# In[ ]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[ ]:


print("Mean Absolute Error = ", mae)


# In[ ]:




