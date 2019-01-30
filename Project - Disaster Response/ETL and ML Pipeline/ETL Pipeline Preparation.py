#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()

# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# merge datasets
df = messages.merge(categories, how='outer', on=['id'])
df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# create a dataframe of the 36 individual category columns
id_series = pd.Series(categories.id.values, index = categories.index)
categories = pd.Series(categories.categories).str.split(pat=';', expand=True)
categories.head()

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = [w[:-2] for w in row.values]
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
categories.head()

# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# drop the original categories column from `df`
df = df.drop('categories', axis=1)
df.head()

# concatenate the original dataframe with the new `categories` dataframe
categories['id'] = id_series
df = df.merge(categories, how='outer', on =['id'])
df.head()

# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# check number of duplicates
df.duplicated().sum()

# drop duplicates
df = df[~df.duplicated()]

# check number of duplicates
df.duplicated().sum()

# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('Messages', engine, index=False)

# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.
