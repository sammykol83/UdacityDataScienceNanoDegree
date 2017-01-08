import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads the csv messages and categories data and combines them to a
    single standard dataframe.

    Args:
        messages_filepath (str): path to messages csv file.
        categories_filepath (str): path to categories csv file.

    Returns:
        DataFrame: The dataframe combining both inputs
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])

    # Split the values in the `categories` column on the `;`
    id_series = pd.Series(categories.id.values, index=categories.index)
    categories = pd.Series(categories.categories).str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [w[:-2] for w in row.values]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    categories['id'] = id_series
    df = df.merge(categories, how='outer', on=['id'])

    return df


def clean_data(df):
    """Cleans the dataframe (removes duplicates).

       Args:
           df (DataFrame): dataframe to clean

       Returns:
           DataFrame: Cleaned DataFrame
    """

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # drop duplicates
    df = df[~df.duplicated()]

    return df

def save_data(df, database_filename):
    """Saves a pandas DataFrame into an SQLlite database

          Args:
              df (DataFrame): dataframe to save
              database_filename (str): filename for SQLlite database

          Returns:
              None
       """
    engine_str = 'sqlite:///' + database_filename
    engine = create_engine(engine_str)
    df.to_sql('InsertTableName', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()