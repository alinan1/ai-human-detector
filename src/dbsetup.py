# dbsetup.py
import sqlite3
import pandas as pd

def clean_df(df):
    df = df.drop_duplicates()  # removes all rows that are copies of eachother
    df = df.dropna(subset=['text']) #removes any rows where "text" col is blank
    df = df.dropna(subset=['generated']) #removes any rows where any "generated" col is blank

    df['text'] = df['text'].str.strip().str.lower() # convert text to lowercase and remove any whitespace
    return df

def setup_database():
    #load csv
    df = pd.read_csv('data/AI_Human.csv')
    
    #clean data
    df = clean_df(df)

    # opens connection to sqlite database file called  ai_human.db'
    conn = sqlite3.connect('database/ai_human.db')

    # saves data as a TABLE inside of my database (ai_human.db) called "texts" (if table already exists, it will be replaced)
    # #save the table into whatever database conn is connected to
    df.to_sql('texts', conn, if_exists='replace', index=False)
 
    conn.close()
    print("Database has been setup!")

if __name__ == "__main__":
    setup_database()