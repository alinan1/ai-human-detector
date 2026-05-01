# visual.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')
    df = pd.read_sql("SELECT * FROM extracted_texts", conn)
    conn.close()
    return df

def _grouped_bars(ax, avg_df, features, under_names, ylabel, title, fmt='.1f'):
    """Helper: draw a grouped bar chart (Human | AI) on the given axes."""
    x     = np.arange(len(features))
    width = 0.35
    colors = ["#3292F3", "#EC985C"]

    bars_h = ax.bar(x - width / 2, avg_df.loc['Human', features],
                    width, label='Human', color=colors[0], zorder=3)
    bars_a = ax.bar(x + width / 2, avg_df.loc['AI Generated', features],
                    width, label='AI Generated', color=colors[1], zorder=3)

    all_vals = list(avg_df.loc['Human', features]) + list(avg_df.loc['AI Generated', features])
    offset   = max(all_vals) * 0.02

    for bar, col in zip([*bars_h, *bars_a], [colors[0]] * len(bars_h) + [colors[1]] * len(bars_a)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f'{bar.get_height():{fmt}}',
                ha='center', va='bottom', fontsize=9,
                color=col, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(under_names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(all_vals) * 1.22)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('#F7F9FC')

def plot_average_comparison(df):
    avg_df = df.groupby('generated')[
        ['word_count', 'sentence_count', 'avg_sentence_length',
         'vocab_diversity', 'punctuation_freq']
    ].mean().rename(index={0.0: 'Human', 1.0: 'AI Generated'})

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Average Data Comparison: Human vs AI Generated',
                 fontsize=14, fontweight='bold', y=1.02)

    _grouped_bars(
        axes[0], avg_df,
        features   = ['word_count', 'sentence_count'],
        under_names = ['Word Count', 'Sentence Count'],
        ylabel     = 'Average Count',
        title      = 'Text Length',
        fmt        = '.0f'
    )

    _grouped_bars(
        axes[1], avg_df,
        features   = ['avg_sentence_length'],
        under_names = ['Average Sentence Length'],
        ylabel     = 'Words per Sentence',
        title      = 'Average Sentence Length\n(Words per Sentence)',
        fmt        = '.1f'
    )

    _grouped_bars(
        axes[2], avg_df,
        features   = ['vocab_diversity', 'punctuation_freq'],
        under_names = ['Vocabulary\nDiversity', 'Punctuation\nFrequency'],
        ylabel     = 'Percentage (%)',
        title      = 'Vocab Diversity & Punctuation Freq\n(Percentages)',
        fmt        = '.1f'
    )

    plt.tight_layout(pad=2.5)
    plt.savefig('visuals/average_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_visualizations():
    os.makedirs('visuals', exist_ok=True)
    df = load_data()
    plot_average_comparison(df)
    print("All visualizations completed")

if __name__ == "__main__":
    run_visualizations()