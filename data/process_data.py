import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data

    Args:   Filepath to messages csv file
            Filepath to categories csv file

    Output: Pandas DataFrame (df) which contains merged csv files
    ....
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.concat([messages, categories], sort=False, axis=1)

    return df

def clean_data(df):
    """
    Clean Data

    Args:   Raw DataFrame (df)
    Output: Cleaned DataFranme (df)
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    for x in row:
        category_colnames.append(x[0:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.head()

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save Data

    Args:   Cleaned DataFrame (df), 
            Destination path for saving the database (.db)
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('df_table', engine, index=False)


def main():
    """
    Main Function

    This is the Main Function which is processing the steps if ETL pipeline.
        - Load Data
        - Clean and preprocess Data
        - Export Data to SQLite database file
    """

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()