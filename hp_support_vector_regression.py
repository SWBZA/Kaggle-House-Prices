#!/usr/bin/env python
# coding: utf-8

# # Support Vector Regression (SVR)

# ## Importing the libraries

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qgrid

#  Make sure all print() lines are printed, not just the last one
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Make sure matplotlib charts and graphs are displayed in the cell outputs
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the dataset
# >
# > This notebook expects cleaned data
# > See "EDA.ipynb" - used to clean data

# In[2]:


dataset = pd.read_csv("cleaned_data.csv")
# dataset.info()


# In[3]:


# dataset.iloc[0:10, 0]


# In[4]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[5]:


y = y.reshape(len(y), 1)


# In[6]:


np.shape(X)


# In[7]:


np.shape(y)


# ## Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:


# np.shape(X_train)
# # np.shape(X_test)
# # np.shape(y_train)
# # np.shape(y_test)


# ## Feature Scaling

# In[10]:


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


# ## Training the SVR model on the Training set

# In[11]:


type(y_train)


# In[12]:


y_train = y_train.ravel()  #   Note the .ravel() to convert into one dimension


# In[13]:


# np.shape(y_train)


# In[15]:


from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(X_train, y_train)

type(y_train)


# ## Predicting the Test set results

# In[16]:


y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance
# >######################################################################################################
# >######################################################################################################

# In[18]:


# %%script  
# from sklearn.metrics import r2_score

# r2_score(y_test, y_pred)


# In[19]:


get_ipython().run_line_magic('lsmagic', '')


# >######################################################################################################
# >######################################################################################################

# # Make competition submission

# In[20]:


# Import data
data = pd.read_csv("hp_test.csv")
# data.info()


# In[21]:


# Keep only the features as decided through the EDA
keep_numerical = [
    "GrLivArea",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
]
keep_categorical = [
    "OverallQual",
    "Neighborhood",
    "GarageCars",
    "ExterQual",
    "BsmtQual",
    "KitchenQual",
    "FullBath",
    "GarageFinish",
    "FireplaceQu",
    "Foundation",
    "GarageType",
]
data_subset = data[keep_categorical + keep_numerical]
# data_subset.info()


# In[22]:


# qgrid_widget = qgrid.show_grid(pd.DataFrame(data_subset),
#                                show_toolbar=True,
#                               grid_options={'forceFitColumns': False}  #   Many columns in dataframe become unusable for filter without this
#                               )
# qgrid_widget


# In[23]:


# Deal with missing data
data_subset.drop(columns="FireplaceQu", inplace=True)


# In[24]:


# Encode categorical features
data_subset_enc = pd.get_dummies(data_subset)


# In[25]:


# qgrid_widget = qgrid.show_grid(pd.DataFrame(data_subset_enc),
#                                show_toolbar=True,
#                               grid_options={'forceFitColumns': False}  #   Many columns in dataframe become unusable for filter without this
#                               )
# qgrid_widget


# In[26]:


# data_subset_enc.info()


# In[27]:


#  Now prepare for competition submission
#####  Confirm why first column is dropped ##########
X_competition = data_subset_enc.iloc[:, :].values


# In[28]:


np.shape(X_competition)


# In[29]:


# Now only have numeric data, so impute missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X_competition)
X_competition = imputer.transform(X_competition)
# np.shape(X_competition)


# In[30]:


y_competition = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_competition)))
# np.shape(y_competition)


# In[31]:


# dataset.iloc[0:10, 0]


# In[32]:


# Concatenate prediction and passenger ID columns
# submission = pd.DataFrame(pd.concat([dataset.iloc[:, 0],  y_competition.iloc[:, :]], axis=1))
submission = pd.DataFrame(y_competition)


# In[33]:


submission.columns = ['SalePrice']
IdColumn = pd.read_csv("hp_test.csv")
# submission = pd.concat(IdColumn["Id"], submission["SalePrice"])
# submission
submission = pd.concat([IdColumn['Id'], submission['SalePrice']], axis=1)
#print(submission)


# In[34]:


# Write to file
submission.to_csv("hp_svr_rbf.csv", index=False)


# In[ ]:




