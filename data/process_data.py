import sys
import pandas as pd
import numpy as np
import sqlite3

#1.load&extract data of interest
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    # merge datasets
    df = messages.merge(categories,on=['id'])
    return df
#someting reusable for other scripts:
#load_data('disaster_messages.csv','disaster_categories.csv')

#2.cleaning the loaded data and return DataFrame (one-hot data:message types by message samples)
def clean_data(df):
    categories = pd.Series(df['categories']).str.split(';',expand=True)
    row = categories.loc[[0]]
    category_colnames = row.apply(lambda x: x.str.split('-').str.get(0))
    categories.columns = category_colnames.loc[0]
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)#注意如果已是字符串二次转化会出错
        # convert column from string to numeric
        categories[column] = categories[column].fillna(0).astype('int')
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates('original',inplace=True)
    return df

#3.save data as SQL file
def save_data(df, database_filename, database_filepath):
    conn=sqlite3.connect(database_filepath)
    df.to_sql(database_filename, con=conn, if_exists='replace',index=False)
#someting reusable for other scripts:
#save_data(df,'disaster_data.db')

#4.given csv files and return the processed one
def main(messages_filepath, categories_filepath, database_filename,database_filepath):
    #messages_filepath, categories_filepath, database_filepath

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filename, database_filepath)
        
    print('Cleaned data saved to database!')
main('data/disaster_messages.csv','data/disaster_categories.csv','disaster_data','disaster_data.db')
