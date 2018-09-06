
# coding: utf-8

# In[41]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt


# Working on Series

# In[10]:


object = Series([5,10,15,20])
print("Direct Printing of Series \n"+str(object))
print("Printing of Series with values \n"+str(object.values))
print("Printing of Series with index \n"+str(object.index))


# Using Numpy Arrays to Series

# In[8]:


data_array = np.array(['a','b','c'])
s = Series(data_array)
print("Conversion of Array into Series : \n"+str(s))

#custom indexing
s = Series(data_array,index=[100,101,102])
print("Series with Custom Index : \n"+str(s))

#using real life ex
revenue = Series([20,80,40,35],index=['ola','uber','grab','gojek'])
print(revenue)
print("Finding Revenue of Ola from above series with index as ola :"+str(revenue['ola']))

print("Finding values which has Revenue greater than equal to 35 :\n"+str(revenue[revenue>=35]))

#use boolean conditions
print("Boolean Condition if 'lyft' present in Series : "+str('lyft' in revenue))

#nan values
index_2 = ['ola','uber','grab','gojek','lyft']
revenue2 = Series(revenue,index_2)
print("Indexing and assigning NaN value for extra index "+str(revenue2))


#isnull and notnull
print("Checking is series contains any null values :\n"+str(pd.isnull(revenue2)))
print("Checking is series contains any notnull values :\n"+str(pd.notnull(revenue2)))

#addition of series (+)
print("Addition of Both Series is :\n"+str(revenue+revenue2))

#assigning names
revenue2.name="Company Revenues"
revenue2.index.name="Comapany Name"
print("Assigning Series after name and index name \n"+str(revenue2))

revenue_dict = revenue.to_dict()
print("Conversion of Series into Dictionary :\n"+str(revenue_dict))


# DataFrames for read from clipboard

# In[ ]:


#example - Revenue of companies

revenue_df = pd.read_clipboard()
print(revenue_df)

#index and columns
print revenue_df.columns
print revenue_df['Rank ']
# #multiple columns

print DataFrame(revenue_df,columns=['Rank ','Name ','Industry '])

#Nan Values
revenue_df2 = DataFrame(revenue_df,columns=['Rank ','Name ','Industry ','Profit'])
print revenue_df2

#head and tail
print revenue_df.head(2)
print revenue_df.tail(2)

#access rows in df
print revenue_df.ix[0] #row 1
print revenue_df.ix[5] #row 6

#assign values to df
#numpy

array1 = np.array([1,2,3,4,5,6])
revenue_df2['Profit'] = array1
print revenue_df2

#series
profits = Series([900,1000],index=[3,5])
revenue_df2['Profit'] = profits

print revenue_df2

#deletion
del revenue_df2['Profit']
print revenue_df2


# Dictonary function to Dataframe 

# In[3]:


#dictionary function to dataframe
sample = {
     'Name':['Vijender','Barrett'],
     'Salary':['$100000','$50000']
}
#
print(sample)
#
sample_df = DataFrame(sample)
print(sample_df)


# Index-Objects

# In[7]:


series1 = Series([10,20,30,40],index=['a','b','c','d'])

index1 = series1.index
print(index1)

print(index1[2:])

#negative indexes
print(index1[-2:])
print(index1[:-2])

print(index1[2:4])

#interesting
#index1[0] = 'e' #TypeError: Index does not support mutable operations


# In[ ]:


Reindexing in pandas


# In[46]:


#create new series series1

cars = Series(['Audi','Merc','BMW'],index=[1,2,3])
print("Default Series : \n"+str(cars))

#creating new indexes using reindex
cars = cars.reindex([1,2,3,4])
print("use of reindex in default Series : \n"+str(cars))

#using fillvalue
cars = cars.reindex([1,2,3,4,5],fill_value='Honda')
print("use of fillvalue in Series : \n"+str(cars))

#using reindex methods => ffill
ranger = range(8)
print(ranger)

cars = cars.reindex(ranger,method="ffill") #forward fill
print("use of ffill in default Series : \n"+str(cars))


# In[71]:


#create new dataframe using randn
df_1 = DataFrame((abs(randn(25))*100).reshape(5,5),index=['a','b','c','d','e'], columns=['c1','c2','c3','c4','c5'])
print("Default DataFrame \n"+str(df_1))

#reindex rows of dataframe
df_2 = df_1.reindex(['a','b','c','d','e','f'])
print("\n Reindexed Rows \n"+str(df_2))

# #reindex columns of dataframe
df_3 = df_2.reindex(columns=['c1','c2','c3','c4','c5','c6'])
print("\n Reindexed Columns \n"+str(df_3))

# #using .ix[] to reindex
df_4 = df_1.ix[['a','b','c','d','e','f'],['c1','c2','c3','c4','c5','c6']]
print("\n Reindexed Both Rows and Columns using ix function \n"+str(df_4))


# In[ ]:


Drop Entries


# In[77]:


cars = Series(['Audi','Merc','BMW'],index=[1,2,3])
cars = cars.drop(1)
print("After Drop function to delete Audi \n"+str(cars))

#dataframes
cars_df = DataFrame(np.arange(9).reshape(3,3),index=['BMW','Audi','Merc'],columns=['rev','pro','exp'])
print("\nDataFrame as \n"+str(cars_df))

cars_df = cars_df.drop('BMW',axis=0)
print("\n After dropping 'BMW' value from dataframe \n"+str(cars_df))

cars_df = cars_df.drop('pro',axis=1)
print("\n After dropping by 'pro' index from dataframe \n"+str(cars_df))


# In[82]:


Handling Null Data in Series


# In[83]:


series1 = Series(['A','B','C','D',np.nan])
print("Default Series as \n"+str(series1))

#validate
print("\n Check for isnull in Series \n"+str(series1.isnull()))
print("\n Check after dropna in Series \n"+str(series1.dropna()))


# Handling Null Data in DataFrame

# In[91]:


df1 = DataFrame([[1,2,3],[5,6,np.nan],[7,np.nan,10],[np.nan,np.nan,np.nan]])
print("Default DataFrame is \n"+str(df1))

print("\n DataFrame after dropping na \n"+str(df1.dropna()))
print("\n DataFrame after dropping where all values as na \n"+str(df1.dropna(how='all')))

#column wise drop6,
print("\n DataFrame after dropping na column wise \n"+str(df1.dropna(axis=1)))

df2 = DataFrame([[1,2,3,np.nan],[4,5,6,7],[8,9,np.nan,np.nan],[12,np.nan,np.nan,np.nan]])
print("\n New Dataframe as \n"+str(df2))

print("\n DataFrame after dropping na with threshold=3 \n"+str(df2.dropna(thresh=3)))
print("\n DataFrame after dropping na with threshold=2 \n"+str(df2.dropna(thresh=2)))

#fillna
print("\n DataFrame after fillna with one value \n"+str(df2.fillna(0)))
print("\n DataFrame after fillna with all values\n"+str(df2.fillna({0:0,1:50,2:100,3:200})))


# In[ ]:


Selecting Moditying Entires


# In[92]:


series1 = Series([100,200,300],index=['A','B','C'])
print("Default Series is \n"+str(series1))

print(series1['A'])
print(series1[['A','B']])

#number indexes
print(series1[0])
print(series1[0:2])

#conditional indexes
print(series1[series1>150])
print(series1[series1==300])

#using df and accesing
df1 = DataFrame(np.arange(9).reshape(3,3),index=['car','bike','cycle'],columns=['A','B','C'])
print("\n Default DataFrame \n"+str(df1))

print(df1['A'])
print(df1[['A','B']])

print("\n Dataframe with value>5 \n"+str(df1>5))

#ix function access
print(df1.ix['bike'])
print(df1.ix[1])


# In[ ]:


Data Alignment


# In[93]:


ser_a = Series([100,200,300],index=['a','b','c'])
ser_b = Series([300,400,500,600],index=['a','b','c','d'])

#sum of series
print("Sum of series\n"+str(ser_a+ser_b))

#dataframe
df1 = DataFrame(np.arange(4).reshape(2,2),columns=['a','b'],index=['car','bike'])
print(df1)
df2 = DataFrame(np.arange(9).reshape(3,3),columns=['a','b','c'],index=['car','bike','cycle'])
print(df2)
print(df1+df2)

#important
df1 = df1.add(df2,fill_value=0)
print(df1)

ser_c = df2.ix[0]
print(df2 - ser_c)


# In[ ]:


Ranking Sorting


# In[94]:


ser1 =Series([500,1000,1500],index=['a','c','b'])
print(ser1)
#sorting by index
print(ser1.sort_index())

#sort by values
print(ser1.sort_values())

print(ser1.rank())


# In[ ]:


Pandas Statistical


# In[123]:


array1 = np.array([[10,np.nan,20],[30,40,np.nan]])
#print(array1)
df1 = DataFrame(array1,index=[1,2],columns=list('ABC'))
print("Default DataFrame \n"+str(df1))

#sum()
print("\n Sum along each column \n"+str(df1.sum()))
print("\n Sum along indexes \n"+str(df1.sum(axis=1)))

print("\n Min in dataframe in column wise\n"+str(df1.min()))
print("\n Max in dataframe in column wise\n"+str(df1.max()))

#idxmax : Return index of first occurrence of maximum over requested axis. NA/null values are excluded
print("\n Max in dataframe with idxmax\n"+str(df1.idxmax()))

#cumsum : Return cumulative sum over a DataFrame or Series axis
print("\n Dataframe with cumsum\n"+str(df1.cumsum()))


print("\n Dataframe with Describe function\n"+str(df1.describe()))


# Unique Values in Series and count of unique

# In[127]:


ser1 = Series(list('abcccaabd'))
print("Outputing unique values in list : "+str(ser1.unique()))
print("\nOutputing count of unique values in list \n"+str(ser1.value_counts()))


# Statiscal Ploting DataFrame

# In[130]:



df2 = DataFrame(abs(randn(9)).reshape(3,3),index=[1,2,3],columns=list('ABC'))
print("\n Default Dataframe \n"+str(df2))

plt.plot(df2)
plt.legend(df2.columns,loc="lower right")
plt.savefig('samplepic.png')
plt.show()

