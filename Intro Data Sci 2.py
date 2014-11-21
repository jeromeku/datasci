# Objectives today:
#
# Basic command line looking at data
# Launch & use basics of iPython Notebook
# Load data as a pandas data frame
# Investigate what type of data is in there (columns, data elements in rows)
# Basic:
#    slicing (by columns, rows),
#    filtering (boolean vectors as data frame indices),
#    aggregating (value_counts, groupby, aggregate)
# Basic plots: scatter, bar, time series
# Where to go for help (StackOverflow)


#First let's look at at our data with the command line:

wc -l 311-service-requests.csv
head -2 311-service-requests.csv
head -1 311-service-requests.csv | grep -o "," | wc -l
head -2 311-service-requests.csv | grep -o "," | wc -l


#We'll come back to this data later, but for now let's open up an iPython session and see what that gives us
# Tip: navigate first to the folder where you'll want your iPython notebook file saved.

ipython notebook


# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


#To get started in the notebook, let's import the python modules we know we'll use, and set a couple environment variables

%pylab inline #inline graphs
import numpy as np
import scipy
import pandas as pd
pd.options.display.mpl_style = 'default'


# describe, unique, groupby, aggregate, pivot_table, stack/unstack, value_counts, merge


names = ['Bob', 'Pete', 'Jim', 'Sarah', 'Jessica', 'Jacqueline']
births = [1234, 432, 2345, 234, 8493, 23]
sex = ['Male','Male','Male','Female','Female','Female']
babies = zip(names, births, sex)
babies

df = pd.DataFrame(data = babies, columns = ['Names', 'Births','Sex'])
df

df.describe()

df.info()

df.sort(['Births'], ascending=False)

df[['Names','Births']].set_index('Names').sort('Births', ascending=False).plot(kind='bar')

df.groupby('Sex').sum() / df['Births'].sum()

df['Births'].sum()

df['Births'].apply(np.sqrt)

df['Births'].map(lambda x : np.sqrt(x))

np.sqrt(df['Births'])

df['Sex'].str.upper()


# OK, so at this point we've seen a super high-level view of some of the things you can do with this data frame object. Let's digress for a few minutes into the key pandas data structures so that we can understand what's going on as we manipulate them.

# Messing around with pandas data structures

# A Series is essentially a vector with named indices, or like a spreadsheet with a single column.
# Probably gets its name from "time series" where the indices are timestamps

# Construction from a 1-dimensional array and a list of indices
s = pd.Series(np.random.randn(5), index=['a','b','c','d','e'])

# Construction from a dict, by default pulls the keys from the dict to be the indices
d = {'a' : 0., 'b' : 1., 'c' : 2.}
s2 = pd.Series(d)

# Construction from a scalar (like the 'recycling rule' in R)
s3 = pd.Series(5, index=[1,2,3,4,5])

# Series can also have names (keep this in mind for later)
s.name = 'Some Data'

s

# Accessing elements of a series

# by slice indexing
s[0]
s[[0,3]]
s[:3]

# by index names
s['a']
s[['a','e']]

# by boolean vector

s[s > 0]
s[s == max(s)] #compare with max(s) or s.max()


# example
s[['a','e']]

# Data Frames
#
# These are essentially combinations of Series which have the same shared index.
# Effectively equivalent to spreadsheets where the index is the row labels, and each series is a column.
# Each column can be a different data type (numeric, strings, etc.)
#
# Just as the lowly spreadsheet is one of the most common formats for data, the data frame is what you'll likely use most.

d = {'s-one' : s, 's-two': s2}
df = pd.DataFrame(d)
df


# With that, let's go back to the dataset we originally intended to look at.

data = pd.read_csv('/Users/bgardner6/Dropbox/dev/introdatasci/311-service-requests.csv', parse_dates=['Created Date'])

data.describe()

data.info()

data['Complaint Type'].value_counts()

# HEATING complaints are evidently a big deal in this dataset. Let's look at that subset.
heat = data[['Created Date', 'Complaint Type']][data['Complaint Type'] == 'HEATING']

heat.set_index('Created Date')

heat.info()

heat['day'] = heat['Created Date'].apply(lambda x : x.day)
heat['day'].hist(bins=len(heat['day'].unique()))

data['Created Date'].max()

# Noise is also common, and looked like it might be split into different types. Let's investigate.
noise = data[['Created Date', 'Complaint Type']][data['Complaint Type'].str.lower().str.contains('noise', na=False)]

len(noise)

noise['Complaint Type'].value_counts().plot(kind='bar')

noise_by_hour = noise.set_index('Created Date').resample('H', how='count')
noise_by_hour

noise['hour'] = noise['Created Date'].apply(lambda x: x.hour)

noise['hour'].hist(bins=24)

# How about some of the other columns in the dataset?

data[['Complaint Type','Created Date','Agency','Borough', 'Bridge Highway Name']][:5]

set(data['Bridge Highway Name'])
data['Bridge Highway Name'][data['Bridge Highway Name'].notnull()]
data['Bridge Highway Name'].value_counts()
data['Bridge Highway Name'].value_counts()[:10].plot(kind='bar')

# What about latitude and longitude? What fun can we have with that?
plot(data['Longitude'], data['Latitude'], '.', color="purple")


# Or how about Agency?

agency = data[['Agency','Agency Name','Latitude','Longitude','Created Date']]
lookup = agency[['Agency','Agency Name']].groupby('Agency').agg('first').reset_index()
lookup
groups = agency.groupby('Agency')
top_agencies = set(agency['Agency'].value_counts()[10:15].index)
top_agencies

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    if name in top_agencies:
        ax.plot(group.Longitude, group.Latitude, marker='.', linestyle='', ms=12, label=name)
ax.legend()

plt.show()

for name in top_agencies: print name
