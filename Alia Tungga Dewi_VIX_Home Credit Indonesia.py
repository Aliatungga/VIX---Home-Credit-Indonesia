#!/usr/bin/env python
# coding: utf-8

# # LIBRARIES

# In[141]:


# Data manipulation
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 99)

# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

# File system manangement
import os

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# # IMPORTING DATA

# In[142]:


df_train = pd.read_csv('application_train.csv')
df_test = pd.read_csv('application_test.csv')
print("Train Data Shape b4 adding target col : ",df_train.shape)
print("Test Data Shape b4 adding target col : ",df_test.shape)


# In[143]:


df_train["source"] = "train"
df_test["source"] = "test"
print("Train Data Shape aftr adding target col : ",df_train.shape)
print("Test Data Shape aftr adding target col : ",df_test.shape)


# In[144]:


df = pd.concat([df_train,df_test])


# # EXPLORING DATA

# We can easily convert the dataset into a pandas dataframe to perform exploratory data analysis. Simply pass in the dataset.data as an argument to pd.DataFrame(). We can view the first 5 rows in the dataset using head() function.

# In[145]:


df.shape


# find the dimension of given data

# In[146]:


df.info()


# In[147]:


df.columns


# We can check the datatype of each column using dtypes to make sure every column has numeric datatype. If a column has different datatype such as string or character, we need to map that column to a numeric datatype such as integer or float. For this dataset, luckily there is no such column.

# In[148]:


df.dtypes


# In[149]:


df.sample()


# In[150]:


df.SK_ID_CURR.nunique()


# In[151]:


df.head()


# # REMOVING COLUMNS

# In[152]:


lst=[]
lst=df.columns
row=df.shape[0]
cols=[]
len(cols)


# In[153]:


# Removing columns which has more than 60% of NA Values .
[cols.append(i) for i in lst if df[i].isnull().sum()/row*100 > 60]


# In[154]:


len(cols)


# In[155]:


cols


# In[156]:


cols_to_drop = [ 
               #has more than 60% of NA Values .
                'OWN_CAR_AGE',
                'YEARS_BUILD_AVG',
                'COMMONAREA_AVG',
                'FLOORSMIN_AVG',
                'LIVINGAPARTMENTS_AVG',
                'NONLIVINGAPARTMENTS_AVG',
                'YEARS_BUILD_MODE',
                'COMMONAREA_MODE',
                'FLOORSMIN_MODE',
                'LIVINGAPARTMENTS_MODE',
                'NONLIVINGAPARTMENTS_MODE',
                'YEARS_BUILD_MEDI',
                'COMMONAREA_MEDI',
                'FLOORSMIN_MEDI',
                'LIVINGAPARTMENTS_MEDI',
                'NONLIVINGAPARTMENTS_MEDI',
                'FONDKAPREMONT_MODE',

                #all null / constant / others
                'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_3',
                'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11',
                'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19', 
                'FLAG_DOCUMENT_20', 
                'FLAG_DOCUMENT_21',
                'AMT_REQ_CREDIT_BUREAU_HOUR', 
                'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 
                'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 
                'AMT_REQ_CREDIT_BUREAU_YEAR'
                
]


# In[157]:


data = df.drop(cols_to_drop, axis=1)


# In[158]:


data.shape


# In[159]:


data.head()


# In[160]:


data.isnull().any()


# In[161]:


data.isna().sum()


# In[162]:


data.duplicated().sum()


# In[163]:


data.describe(include='all').T


# # EXPLORATORY DATA ANALYSIS

# ## Correlation Check

# Finding correlation between attributes is a highly useful way to check for patterns in the dataset.

# In[164]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# data.corr
corrmat=data.corr()
plt.figure(figsize=(22,19))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()


# In[165]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# if there are pairs of features that have a high correlation, only one will be taken. The correlation value that is used as a benchmark as a high correlation is uncertain, generally the number 0.7 is used.

# In[166]:


corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop_hicorr = [column for column in upper.columns if any(upper[column] > 0.7)]
to_drop_hicorr


# ## Target Variable Distribution

# In[167]:


data.head()


# In[168]:


pd.options.display.float_format = '{:,.2f}'.format
x = data['TARGET'].value_counts()
print('The number of defaults in the target variable is {} whereas the number of repayments is {}'.format(x.loc[0],x.loc[1]))
x.plot.pie()


# ## Missing Value Cheking

# Sometimes, in a dataset we will have missing values such as NaN or empty string in a cell. We need to take care of these missing values so that our machine learning model doesn’t break. To handle missing values, there are three approaches followed. 
# When it comes time to build our machine learning models, we will have to fill in these missing values (known as imputation). In later work, we will use models such as XGBoost that can handle missing values with no need for imputation. Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns will be helpful to our model. Therefore, we will keep all of the columns for now.

# In[169]:


check_missing = data.isnull().sum() * 100 / data.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)


# In[170]:


data.drop(to_drop_hicorr, axis=1, inplace=True)


# In[171]:


pd.options.display.float_format = '{:,.2f}'.format
df_missing_val = pd.DataFrame(columns = ['column','number of missing vals'])

for column in data.columns:
    missing_val = data[column].isnull().sum()
    x = {'column':column,'number of missing vals':missing_val}
    df_missing_val= df_missing_val.append(x, ignore_index=True)
df_missing_val = df_missing_val[df_missing_val['number of missing vals'] !=0]
df_missing_val = df_missing_val.sort_values(by='number of missing vals',ascending=False)
print('Below is a table showing the number of missing values per column. The numbers are quite significant for some features. However, we shall deal withbthis at a later stage')

df_missing_val.head(10)


# ## Corelation without Variable

# In[172]:


cor_target = abs(data.corr())
#Selecting highly correlated features
relevant_features = cor_target[cor_target<0.3]
relevant_features


# In[173]:


type(relevant_features)
relevant_features.items


# ## Checking Categorical Features

# In[174]:


data.select_dtypes(include='object').nunique()


# Most of the categorical variables have a relatively small number of unique entries. We will need to find a way to deal with these categorical variables!
# 

# In[175]:


data.select_dtypes(exclude='object').nunique()


# In[176]:


for col in data.select_dtypes(include='object').columns.tolist():
    print(data[col].value_counts(normalize=True)*100)
    print('\n')


# # DEALING WITH CATEGORICAL VARIABLES

# ## One Hot Encoding

# One-hot encoding: create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.

# In[177]:


train_data = pd.get_dummies(data)
test_data = pd.get_dummies(data)
print('The number of training features and variables is: {}'.format(train_data.shape))
print('The number of test features and variables is: {}'.format(test_data.shape))


# ## Label Encoding

# In[178]:


le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train_data:
    if train_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(train_data[col])
            # Transform both training and testing data
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# ## Aligning the variables in the test and train datasets

# There need to be the same features (columns) in both the training and testing data. One-hot encoding has created more columns in the training data because there were some categorical variables with categories not represented in the testing data. To remove the columns in the training data that are not in the testing data, we need to align the dataframes. First we extract the target column from the training data (because this is not in the testing data but we need to keep this information). When we do the align, we must make sure to set axis = 1 to align the dataframes based on the columns and not on the rows!

# In[179]:


target_col = train_data['TARGET']
train_data,test_data = train_data.align(test_data,join='inner',axis=1)
train_data['TARGET'] = target_col
print('The number of training features and variables after alignment is: {}'.format(train_data.shape))
print('The number of test features and variables after alignment is: {}'.format(test_data.shape))


# The training and testing datasets now have the same features which is required for machine learning. The number of features has grown significantly due to one-hot encoding. At some point we probably will want to try dimensionality reduction (removing features that are not relevant) to reduce the size of the datasets.

# # Modeling

# In[180]:


target_col.shape


# In[181]:


target_col.head()


# In[182]:


from sklearn.model_selection import train_test_split
train_test_split(train_data,target_col)


# In[183]:


x_train, x_test, y_train, y_test = train_test_split(train_data,target_col)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[184]:


x_train, x_test, y_train, y_test = train_test_split(train_data,target_col, train_size=0.8, random_state=20)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[185]:


scaler = MinMaxScaler().fit(X)
scaled_X = scaler.transform(X)


# In[186]:


seed      = 42
test_size = 0.20

X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = test_size, random_state = seed)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[187]:


ftr_app = train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
target_app = train_data['TARGET']

train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size=0.3, random_state=2020)
train_x.shape, valid_x.shape

