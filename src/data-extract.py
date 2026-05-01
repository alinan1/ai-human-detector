# feature-extract.py
import sqlite3
import pandas as pd
import re

# Extracts numerical features from each text entry
def extract_data(text):
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))

    # Split text into sentences using punctuation marks
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    sentence_count = len(sentences)

    # Average number of words per sentence
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Ratio of unique words to total words
    vocab_diversity = (unique_words / max(word_count, 1)) * 100

    # Finds punctuation characters such as ., !, ?, commas, quotes, etc.
    total_pun_chars = re.findall(r'[^\w\s]', text)

    # Ratio of punctuation to total text length; higher values mean more punctuation usage
    punctuation_freq = (len(total_pun_chars) / max(len(text), 1)) * 100

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'vocab_diversity': vocab_diversity,
        'punctuation_freq': punctuation_freq
    }

def apply_data_extraction():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')

    # Read original cleaned text data
    df = pd.read_sql("SELECT * FROM texts", conn)

    # Apply feature extraction to every text entry
    extracted = df['text'].apply(extract_data)

    # Convert extracted dictionaries into a dataframe
    extracted_df = pd.DataFrame(extracted.tolist())

    # Combine original text/label with extracted features
    final_df = pd.concat([df[['text', 'generated']], extracted_df], axis=1)

    # Save final feature table into database
    final_df.to_sql('extracted_texts', conn, if_exists='replace', index=False)

    print("Data has been extracted and saved to 'extracted_texts' table!")
    print(final_df.head())

    conn.close()

if __name__ == "__main__":
    apply_data_extraction()