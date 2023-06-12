# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:41:15 2023

@author: vansh
"""

# importing all necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# for training and testing set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn import tree
# from sklearn.metrics import accuracy_score,confusion_matrix

df1=pd.read_csv("Bengaluru_House_Data.csv")
pd.set_option('display.max_columns', 500)
df1.head()


df1.groupby('area_type')['area_type'].agg('count')

# removing unwanted columns
unwanted_columns=['area_type','society','balcony','availability']
df2=df1.drop(unwanted_columns,axis='columns')

#------ data cleaning------------------

# checking for null values
df2.isnull().sum()

# removing null values
df3=df2.dropna()
df3.isnull().sum()

# analyzing diffrent featuers

   # size feature
df3['size'].unique()

# here we have multiple same values with different name such as 4 bhk-4 bedroom to remmove this creating a new column "bhk"
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0])) #here we split the values as insividual tokens and access the first token(which is the no of bhk) as split fun. gives us a string as output we convert it to integer
# droping size feature
df4=df3.drop(['size'],axis="columns")

     
    # exploring total_sqft feature
df4.total_sqft.unique()

# here we can observe that all the values are not in float some are in range format and other structure
# we use is float function to check the float values in the total_sqft
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True       
df4[~df4['total_sqft'].apply(is_float)].head(10)


#function to convert range values to mean of the range
def convert(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df5=df4.copy()
df5['total_sqft']=df5.total_sqft.apply(convert)
df5.loc[410]

# finding and dropning null values from totalsqft
df5.isnull().sum()
df6=df5.copy()
df6=df6.dropna()
df6.isnull().sum()

# creating a new column(feature) price_per_sqft
df6['price_per_sqft']=df6['price']*100000/df6['total_sqft']
df6.head()


# analyzing location feature
len(df6.location.unique())
# here location is catogorical  data where there are 1300 different values 

# since we have many values under location we try to reduce the number of values

# removing extra spaces from start or end of the values
df6.location=df6.location.apply(lambda x: x.strip())
location_stats= df6.groupby('location')['location'].agg('count').sort_values(ascending=False)

# from location_stats we understand that thee are many values which have one data point 
# hence we can create a function where all the data points less then say 10 data points comes under the catogory of other

# finding the values less then 10 data points
len(location_stats[location_stats<=10])
location_stats_less_then_10=location_stats[location_stats<=10]

df6.location=df6.location.apply(lambda x: 'other' if x in location_stats_less_then_10 else x)
len(df6.location.unique())


# outlier detection and removal
df7=df6[~(df6.total_sqft/df6.bhk<300)] #removing unreasonable values by considering each room should be atleast 300 sqft less than that is invalid or unreasonable
 
df7.price_per_sqft.describe()

# to remove price per sqft   per location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df8=remove_pps_outliers(df7)

# ploting a scatter plot for 2bhk and 3bhk house having same sqft
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    # matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()    
plot_scatter_chart(df8,"Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df8)
# df8 = df7.copy()
df8.shape


# price per sqft
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

# exploring bathroom featuers
sns.countplot(x=df8.bath)
# here we consider any house with bath room 2 greater that no. of bhk is an outlier 
df8[df8.bath<df8.bath+2]
df9=df8[df8.bath<df8.bath+2]


# visulaization
# to identify impotant featuers for the ML
# location vs price
sns.boxplot(x=df9['location'],y=df9['price'])

# total_sqft vs price
sns.regplot(x=df9['total_sqft'],y=df9['price'],fit_reg=False)
# bath vs price
sns.boxplot(x=df9['bath'],y=df9['price'])
# bhk vs price
sns.boxplot(x=df9['bhk'],y=df9['price'])

df10=df9.drop(["price_per_sqft"],axis="columns")


# -----------------model building-------------------------
# dummies=pd.get_dummies(df10.location)
# df11=pd.concat([df10,dummies])
# df11=pd.get_dummies(df10,drop_first=True)
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11 = df11.drop('location',axis='columns')
x = df11.drop(['price'],axis='columns')
y=df11.price
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# linear  regression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# Use K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)

from sklearn.model_selection import GridSearchCV




# Find best model using GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)

# exporting the model
import pickle
with open('houseing_price.pickle','wb') as f:
    pickle.dump(lr_clf,f)
    
    
    
import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))