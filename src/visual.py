# visual.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data():
    conn = sqlite3.connect('database/ai_human.db') # connect to database
    df = pd.read_sql("SELECT * FROM extracted_texts", conn) # read the data from the "extracted_texts" table in the database into a pandas dataframe
    conn.close() # close the connection to the database bc we no longer need it after loading the data
    return df # return the dataframe containing the data from the database

def _save(fig, filename):
    os.makedirs('visuals', exist_ok=True)
    fig.savefig(f'visuals/{filename}', dpi=150, bbox_inches='tight')
    plt.close(fig)

# Helper function to create grouped bar charts for the average comparison of features between human and AI generated texts
def create_bars(ax, avg_df, features, under_names, ylabel, title, fmt='.1f'):
    
    #---Bar positions and styling---
    x = np.arange(len(features)) # the x positions for the bars, based on the number of features being compared
    width = 0.3 # the width of each bar
    colors = ["#4099F2", "#DA5D5D"] # colors for human and AI bars

    #---Create bars for human and AI generated texts---
    bars_human = ax.bar(
        x - width / 2, # shift human bars to the left
        avg_df.loc['Human', features], # give me values for the human generated texts for the specified features; these values determine the height of the bars
        width, # width of the bars
        label='Human', # label for the legend
        color=colors[0],
        zorder=3 # zorder=3 ensures bars are above gridlines
    )

    bars_ai = ax.bar(
        x + width / 2, # shift AI bars to the right so they appear next to human bars
        avg_df.loc['AI Generated', features], # values for AI generated texts for the specified features
        width,
        label='AI Generated',
        color=colors[1],
        zorder=3 
    )

    # ---Add labels on top of each bar---
    # combine all values from both human and AI bars to determine the maximum value for setting the offset for labels above the bars
    max_height = list(avg_df.loc['Human', features]) + list(avg_df.loc['AI Generated', features])
    offset = max(max_height) * 0.02 # multiply the maximum height by a small percentage to get an appropriate offset for labels above the bars

    # [*bars_human, *bars_ai] unpacks both bar groups into one list so we can add labels to every bar
    # [colors[0]] * len(bars_human) creates one matching color for each human bar, and same idea for AI bars
    for bar, col in zip([*bars_human, *bars_ai], [colors[0]] * len(bars_human) + [colors[1]] * len(bars_ai)):
        # ax.text() adds text labels on top of each bar
        # x position = center of bar; y position = bar height + offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f'{bar.get_height():{fmt}}', # formats the height of the bar, such as .1f for one decimal place
            ha='center',
            va='bottom',
            fontsize=8,
            color=col,
            fontweight='bold'
        )

    ax.set_xticks(x) # tells matplotlib where to place the x-axis ticks
    ax.set_xticklabels(under_names, fontsize=12) # sets x-axis labels to descriptive feature names
    ax.set_ylabel(ylabel, fontsize=12) # sets the y-axis label
    ax.set_ylim(0, max(max_height) * 1.5) # sets y-axis limit higher than tallest bar so labels do not get cut off
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10) # sets subplot title
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0) # adds grid lines for readability
    ax.set_axisbelow(True) # keeps grid behind bars
    ax.set_facecolor('#F7F9FC') # background color for subplot
    ax.legend(fontsize=11) # legend differentiates human and AI bars

# create 2x 4 grid of all 8 features
def plot_all_features(avg_df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Engineered Features: Human vs AI Generated',
                 fontsize=18, fontweight='bold', y=1.02)

    graph = [
        (axes[0, 0], ['word_count'], ['Word Count'], 'Average Words', 'Word Count', '.0f'),
        (axes[0, 1], ['sentence_count'], ['Sentence Count'], 'Average Sentences', 'Sentence Count', '.0f'),
        (axes[0, 2], ['avg_sentence_length'], ['Avg Sentence Length'], 'Words per Sentence', 'Average Sentence Length', '.1f'),
        (axes[0, 3], ['vocab_diversity'], ['Vocabulary Diversity'], 'Percentage (%) of Unique Words', 'Vocabulary Diversity', '.1f'),
        (axes[1, 0], ['punctuation_freq'], ['Punctuation Frequency'], 'Percentage (%) of Text \n Made up of Punctuation Marks', 'Punctuation Frequency', '.1f'),
        (axes[1, 1], ['avg_word_length'], ['Average Word Length'], 'Characters', 'Average Word Length', '.2f'),
        (axes[1, 2], ['stopword_ratio'], ['Stopword Percentage'], 'Percentage (%) of Stopwords', 'Stopword Percentage', '.1f'),
        (axes[1, 3], ['paragraph_count'], ['Paragraph Count'], 'Average # of Paragraphs', 'Paragraph Count', '.1f')
    ]

    for ax, features, names, ylabel, title, fmt in graph:
        create_bars(ax, avg_df, features, names, ylabel, title, fmt=fmt)

    plt.tight_layout(pad=3)
    _save(fig, 'engineered_features.png')

# -------- MODEL ACCURACY GRAPH --------
def plot_modaccuracy(results):

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#F7F9FC')

    modelX = list(results.keys()) # get the model names from the results dictionary, which are the keys of the dictionary
    accuracyY = list(results.values()) # get the accuracy values from the results dictionary, which are the values of the dictionary
    # dictionaries are the results of the different models
    # look like: {"Logistic Regression": 0.85, "Decision Tree": 0.80, "Random Forest": 0.90}

    # create bar standard with model names on x-axis, accuracy values on y-axis; color each bar differently for visual distinction; zorder=3 ensures bars are above gridlines
    bars = ax.bar(modelX, accuracyY, color=["#4099F2", "#DA5D5D", "#6AD078"], zorder=3) #

    # add labels on top of each bar to show the exact accuracy value; format to 3 decimal places
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # set title, y-axis label, and y-axis limits
    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05) # set y-axis limit slightly above 1 to accommodate labels above bars

    plt.tight_layout() # tight_layout adjusts spacing to prevent overlap of elements in the plot
    _save(fig, 'model_accuracy.png')


# -------- FEATURE IMPORTANCE (RANDOM FOREST) --------
def plot_fi(model, feature_names):

    # feature importance data frame has two columns:
    # "Feature" which lists the names of the features, 
    # "Importance" which lists the importance scores for each feature as determined by the random forest model's feature_importances_ attribute; 
    ft_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False) # false - most to least important

    fig, ax = plt.subplots(figsize=(10, 7)) #figsize determines the size of the plot; ax is the subplot object we will use to create the bar chart
    fig.patch.set_facecolor("#F7FCFB") 
    ax.set_facecolor("#F7FBFC")

    ax.bar(ft_df['Feature'], ft_df['Importance'], color="#4099F2", zorder=3)

    # add labels on top of each bar 
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,   
            bar.get_height() + 0.005,            
            f'{bar.get_height():.3f}',           
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    plt.xticks(rotation=75) # rotate so no overlap
    ax.set_title("Importance (random forest)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Feature/data")
    
    plt.tight_layout()
    _save(fig, 'featureimportance_randomforest.png')

# -------- PLOT THESE MODEL GRAPHS --------     
def plot_modelgraph(df, features):
    X = df[features]
    y = df['generated']
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
        test_size=0.2,
        random_state=42
    )

    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(random_state=42), "Random Forest")
    ]

    #results dictionary will store the accuracy of each model, with the model name as the key and the accuracy score as the value\
    result_accuracy = {}
    for model, name in models:
        model.fit(X_train, y_train) # train the model on the training data (X_train and y_train)
        predictions = model.predict(X_test) # use the trained model to make predictions on the test data (X_test)
        accuracy = accuracy_score(y_test, predictions) # calculate the accuracy of the model's predictions by comparing them to the true labels (y_test) using the accuracy_score function from sklearn.metrics
        result_accuracy[name] = accuracy # store the accuracy score in the result_accuracy dictionary with the model name as the key

        if name == "Random Forest":
            plot_fi(model, features)

    plot_modaccuracy(result_accuracy)


def run_visualizations():
    os.makedirs('visuals', exist_ok=True)
    df = load_data()

    allfeatures = ['word_count', 'sentence_count','avg_sentence_length', 'vocab_diversity', 'punctuation_freq','avg_word_length', 'stopword_ratio','paragraph_count']

    # group the dataframe by the 'generated' column so we can calculate average feature values for Human and AI separately
    avg_df = df.groupby('generated')[allfeatures].mean().rename(index={0.0: 'Human', 1.0: 'AI Generated'})

    plot_all_features(avg_df)
    plot_modelgraph(df, allfeatures)
    print("All visualizations completed")

if __name__ == "__main__":
    run_visualizations()