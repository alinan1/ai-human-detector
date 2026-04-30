# feature-extract.py
import sqlite3
import pandas as pd
import re

# function to extract features from text
def extract_features(text):
    words = text.split() # splits txt into list of words
    unique_words = len(set(words)) # counts amount of distinct words --> set is 
    sentences = re.split(r'[.!?]+', text) #splits text at every "." "?" "!" which indicates sentance
    sentences = [s for s in sentences if s.strip()]  # remove empty strings

    # word cont is len (count) of words
    # ai usually generates more consistent legnth vs humans varying
    word_count = len(words)

    # sentances count is just the len (count) of sentances
    # ai usually write more consistent AMOUNT of sentacnes vs humans write more varied AMOUNT of sentances
    sentence_count = len(sentences)

    # calculates average # of words per sentance 
    # ai usually writes more consistent sentance LENGTHS VS humans usually write varied sentance LENGTHS
    avg_sentence_length = word_count / max(sentence_count, 1)

    # calculates ratio/score of unique words : total words (closer to 1.0 --> more diverse)
    # ai usually writes more formulaic and/or repeats vocab (lower diversity) VS humans writing more varied vocab (higher diversity)
    vocab_diversity = unique_words / max(word_count, 1) 

    # identifies how much of text is made up of punctuation characters 
    # ai tends to use punctuatoin more grammatically (typically less) VS humans use it more often since theyre often using more expression characters like (! or ?) etc...
    total_pun_chars = re.findall(r'[^\w\s]', text)
    punctuation_freq = len(total_pun_chars) / max(len(text), 1)

    # return all features as a dictionary
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'vocab_diversity': vocab_diversity,
        'punctuation_freq': punctuation_freq
    }

# applies data extraction to the database and saves results back to a new table
# saves as new table because never want to overwrite original data, always want to be able to go back to it if needed
def data_extraction():
    #open connection to database 
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')
    #read all data from "texts" table into a dataframe
    df = pd.read_sql("SELECT * FROM texts", conn)

    # run extract_data function on every row in 'text' column to retrieve features for each text entry 
    extracted = df['text'].apply(extract_features)

    # save extracted data as a new dataframe 
    # make each feature a column and each row is _said_ extracted feature for each text entry
    extracted_df = pd.DataFrame(extract_features.tolist())

    # ties together the original text and generated label (ai or not ) with the extracted features into one final dataframe
    # table result: text, generated, word_count, sentence_count, avg_sentence_length, vocab_diversity, punctuation_freq
    final_df = pd.concat([df[['text', 'generated']], extracted_df], axis=1) #axis 1 means to concatenate horizontally

    # save this table into the database as a new table called "extracted_texts" (if table already exists, it will be replaced)
    final_df.to_sql('extracted_texts', conn, if_exists='replace', index=False)

    print("data has been extracted and saved to 'extracted_texts' table in the database!")
    print(final_df.head()) # print first 5 rows of final dataframe to verify it looks correct

    conn.close()

if __name__ == "__main__":
    data_extraction()