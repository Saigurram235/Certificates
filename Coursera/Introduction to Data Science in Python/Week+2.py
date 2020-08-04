
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure

# In[4]:


import pandas as pd
get_ipython().magic('pinfo pd.Series')


# In[5]:


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


# In[6]:


numbers = [1, 2, 3]
pd.Series(numbers)


# In[7]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[8]:


numbers = [1, 2, None]
pd.Series(numbers)


# In[9]:


import numpy as np
np.nan == None


# In[10]:


np.nan == np.nan


# In[11]:


np.isnan(np.nan)


# In[12]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[13]:


s.index


# In[14]:


s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[15]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# # Querying a Series

# In[16]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[17]:


s.iloc[3]


# In[18]:


s.loc['Golf']


# In[19]:


s[3]


# In[20]:


s['Golf']


# In[2]:


sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[3]:


s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead


# In[23]:


s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[24]:


total = 0
for item in s:
    total+=item
print(total)


# In[25]:


import numpy as np

total = np.sum(s)
print(total)


# In[26]:


#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[27]:


len(s)


# In[28]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')


# In[29]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')


# In[30]:


s+=2 #adds two to each item in s using broadcasting
s.head()


# In[31]:


for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[32]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[33]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')


# In[34]:


s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[35]:


original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[36]:


original_sports


# In[37]:


cricket_loving_countries


# In[38]:


all_countries


# In[39]:


all_countries.loc['Cricket']


# # The DataFrame Data Structure

# In[40]:


import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[41]:


df.loc['Store 2']


# In[42]:


type(df.loc['Store 2'])


# In[43]:


df.loc['Store 1']


# In[44]:


df.loc['Store 1', 'Cost']


# In[45]:


df.T


# In[46]:


df.T.loc['Cost']


# In[47]:


df['Cost']


# In[20]:


import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[21]:


df.drop('Store 1')


# In[22]:


df.loc['Store 1']['Cost']


# In[23]:


df.loc[:,['Name', 'Cost']]


# In[ ]:





# In[24]:


df


# In[25]:


copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[26]:


get_ipython().magic('pinfo copy_df.drop')


# In[27]:


del copy_df['Name']
copy_df


# In[28]:


df['Location'] = None
df


# # Dataframe Indexing and Loading

# In[9]:


import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])


# In[10]:


costs = max(df['Cost'])
costs


# In[11]:


costs+=2
costs


# In[12]:


df


# In[13]:


get_ipython().system('cat olympics.csv')


# In[14]:


df = pd.read_csv('olympics.csv')
df.head()


# In[ ]:





# In[15]:


df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


# In[ ]:





# In[16]:


df.columns


# In[9]:


import pandas as pd
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# In[18]:


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# In[19]:


c = len(df['Gold'])
c


# # Querying a DataFrame

# In[20]:


df['Gold'] > 0


# In[ ]:





# In[35]:


only_gold = df.where(df['Gold'] > 0)
only_gold.head()


# In[22]:


only_gold['Gold'].count()


# In[26]:


max((df['Gold'] - df['Gold.1'])/df['Gold'].count())


# In[ ]:





# In[24]:


df[df['Gold'] - df['Gold.1']/df['Gold'] + df['Gold.1'] == max(df['Gold'] - df['Gold.1']/df['Gold'] + df['Gold.1'])]


# In[40]:


df['Gold'].count()


# In[41]:


only_gold = only_gold.dropna()
only_gold.head()


# In[42]:


only_gold = df[df['Gold'] > 0]
only_gold.head()


# In[1]:


my_list = [number for i in range(0,6) ]
my_list


# In[43]:


len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])


# In[44]:


df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]


# In[10]:


import pandas as pd
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# # Indexing Dataframes

# In[15]:


df.head()


# In[17]:



df = df.set_index('Gold')
df.head()


# In[47]:


df = df.reset_index()
df.head()


# In[48]:


df = pd.read_csv('census.csv')
df.head()


# In[49]:


df['SUMLEV'].unique()


# In[50]:


df=df[df['SUMLEV'] == 50]
df.head()


# In[51]:


columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()


# In[52]:


df = df.set_index(['STNAME', 'CTYNAME'])
df.head()


# In[53]:


df.loc['Michigan', 'Washtenaw County']


# In[54]:


df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]


# # Missing values

# In[55]:


df = pd.read_csv('log.csv')
df


# In[56]:


get_ipython().magic('pinfo df.fillna')


# In[57]:


df = df.set_index('time')
df = df.sort_index()
df


# In[58]:


df = df.reset_index()
df = df.set_index(['time', 'user'])
df


# In[59]:


df = df.fillna(method='ffill')
df.head()


# In[ ]:




