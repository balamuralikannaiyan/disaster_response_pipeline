# import libraries
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
      function:
      load message and categories data from separate csv files and then merge them using 'id'

      INPUT:
      messages_filepath - the path of disaster_messages.csv file
      categories_filepath -  the path of disaster_categories.csv file

      OUTPUT:
      df - the merged dataset containing messages and categories in one dataframe 
      """
    
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)    
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
    
    """
      function:
      Clean the data from load step to create separate columns for each category and to encode them in each rows

      INPUT:
      df - the merged dataset containing messages and categories in one dataframe 
      
      OUTPUT:
      df - Cleaned dataset where each category is encoded in separate column
      """      
            
    # create a dataframe of the 36 individual category columns by expanding the categories
    category_columns = df['cartegoies'].str.split(";", expand = True)
    # select the first row of the categories dataframe
    first_row = category_columns.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_column_names = list(first_row.apply(lambda x:x[:-2]))
    # rename the columns of `category columns`
    category_columns.columns = category_column_names
    
    for column in category_columns:
    # set each value to be the last character of the string
        category_columns[column] = category_columns[column].apply(lambda x:x[-1])
    # convert column from string to numeric
        category_columns[column] = category_columns[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, category_columns], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    
    """
      function:
      Saves the cleaned data in a database

      INPUT:
      df - the dataset containing messages and categories
      database_filename - name of the database file

      """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('disaster_messages_categories', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()