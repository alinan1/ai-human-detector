# visual.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')
    df = pd.read_sql("SELECT * FROM texts_features", conn)
    conn.close()
    return df

def plot_histograms(df):
    features = ['word_count', 'avg_sentence_length', 'vocab_diversity', 'punctuation_freq']

    for feature in features:
        plt.figure(figsize=(8, 5))
        for label in df['generated'].unique():
            subset = df[df['generated'] == label][feature]
            plt.hist(subset, bins=30, alpha=0.5, label=str(label))
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'visuals/{feature}_histogram.png')
        plt.show()

def plot_averages(df):
    features = ['word_count', 'avg_sentence_length', 'vocab_diversity', 'punctuation_freq']

    avg_df = df.groupby('generated')[features].mean()

    avg_df.T.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Feature Values: AI vs Human')
    plt.xlabel('Feature')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.legend(title='Generated')
    plt.tight_layout()
    plt.savefig('visuals/average_comparison.png')
    plt.show()

def run_visualizations():
    import os
    os.makedirs('visuals', exist_ok=True)

    df = load_data()
    plot_histograms(df)
    plot_averages(df)
    print("All visualizations saved to visuals/ folder")

if __name__ == "__main__":
    run_visualizations()