import sqlite3
import pandas as pd
import re

# Extracts numerical features from each text entry
def extract_data(text):
    words = text.split() #.split() method splits the text into a list of words based on whitespace
    word_count = len(words) #count num of words --> len() returns the number of items in a list (in this case, the number of words)
    unique_words = len(set(words)) # set is a data structure that only keeps unique values

    # Split text into sentences using punctuation marks like ., !, ?, etc. 
    sentences = re.split(r'[.!?]+', text)
    # remove 
    sentences = [s for s in sentences if s.strip()]

    #num of sentances is the amount of sentences in text
    sentence_count = len(sentences)

    # average words per sentance 
    # total num of words divided by total num of sentences 
    # **EXAMPLE***: 200 words / 10 sentences = 20 words per sentence
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Percentage of unique words used in text 
    # take unique words and divide by total word count to find a PROPORTION 
    # FOR EX: if theres 100 words and 50 are unique proportion is 50/100 = 0.5 which means 50% of the words are unique --> multiply by 100 to get percentage
    vocab_diversity = (unique_words / max(word_count, 1)) * 100

    # Finds punctuation characters such as ., !, ?, commas, quotes, etc.
    total_pun_chars = re.findall(r'[^\w\s]', text)

    # percentage of punctuation characters in text
    # take the number of punctuations there are and divide by total number of characters in text to PROPORTION
    # FOR EX: if theres 100 characters and 10 are punctuation proportion is 10/100 = 0.1 which means 10% of the characters are punctuation --> multiply by 100 to get percentage
    # max() prevents division by zero by ensuring the denominator is at least 1 NOT REALLY NECESSARY but just a safety measure
    punctuation_freq = (len(total_pun_chars) / max(len(text), 1)) * 100

    # average number of characters per word
    # EXAMPLE: "utilize" (7) vs "use" (3) → AI avg_word_length will be higher
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1) 
 
    # stopwords are common filler words that do not add much meaning to a sentence, such as "the", "a" "is" etc.
    # percentage of words that are stopwords (filler words like "the", "and", "I")
    stopwords = {'the','a','an','is','in','it','of','and','to','was','that',
                 'for','on','are','with','as','at','be','this','by','from'}
    stopword_ratio = (sum(1 for w in words if w in stopwords) / max(word_count, 1)) * 100
 
    # number of paragraphs in the text
    # paragraphs are separated by one or more blank lines (\n\n)    
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    paragraph_count = max(len(paragraphs), 1) # at least 1 paragraph
 

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'vocab_diversity': vocab_diversity,
        'punctuation_freq': punctuation_freq,
        'avg_word_length'   : avg_word_length,
        'stopword_ratio'    : stopword_ratio,
        'paragraph_count'   : paragraph_count,

    }

def apply_data_extraction():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')

    # read the original text data from the "texts" table in the database into a dataframe
    df = pd.read_sql("SELECT * FROM texts", conn)

    # apply extract_data function to the "text" column of the original dataframe
    # returns a series of dictionaries where each dictionary contains the extracted data/features for a given text entry
    # each dictionary has the following format:
    # 0  {'word_count': 100, 'sentence_count': 5, 'avg_sentence_length': 20.0, 'vocab_diversity': 50.0, 'punctuation_freq': 10.0} .......
    extracted = df['text'].apply(extract_data)

    # now we convert the series of dictionaries into a new dataframe 
    # **column corresponds to a specific feature (word_count, sentence_count, etc.)
    # **row corresponds to a text entry
    # Format of extracted_df will look like this:
    # word_count | sentence_count | avg_sentence_length | vocab_diversity | punctuation_freq
    #       x    |         y      |        z            |       w         | v
    extracted_df = pd.DataFrame(extracted.tolist())

    # Combine original table with the new extracted features into a final dataframe
    # so now final_df will have the following format:
    # text | generated | word_count | sentence_count | avg_sentence_length | vocab_diversity | punctuation_freq
    final_df = pd.concat([df[['text', 'generated']], extracted_df], axis=1)

    # Save final feature table into database
    final_df.to_sql('extracted_texts', conn, if_exists='replace', index=False)

    print("Completed and saved to a table called: extracted_texts ")
    print(final_df.head())

    conn.close()

if __name__ == "__main__":
    apply_data_extraction()