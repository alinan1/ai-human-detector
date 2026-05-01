# model.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data from SQLite database
def load_data():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')
    df = pd.read_sql("SELECT * FROM extracted_texts", conn)
    conn.close()
    return df

# Train and evaluate the model
# This function takes a model, its name, training and testing data, fits the model, makes predictions, and prints the accuracy and classification report.
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    # LEARN PATTERS
    # fit: This method is used to train the model on the training data (X_train and y_train). It adjusts the model's parameters to minimize the error between the predicted and actual labels.
    # X_train is the feature data used for training the model and y_train is the corresponding labels (0 for human-generated & 1 for AI-generated)
    model.fit(X_train, y_train) 

    # APPLY LEARNED PATTERNS
    # model.predict: This method is used to make predictions on the test data (X_test) using the trained model.
    predictions = model.predict(X_test) 

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))


def run_models():
    df = load_data()

    # the extracted data features that we will use for training the models
    extracted_data = ['word_count', 'sentence_count', 'avg_sentence_length',
                'vocab_diversity', 'punctuation_freq']

    X = df[extracted_data] # input features for the model
    y = df['generated'] # target variable (0 for human-generated, 1 for AI-generated)

    # Different test sizes (30%, 20%, 10%)
    # 0.3 = 30% for testing and 70% for training
    # 0.2 = 20% for testing and 80% for training
    # 0.1 = 10% for testing and 90% for training.
    test_sizes = [0.3, 0.2, 0.1]

    # We will evaluate three different models: Logistic Regression, Decision Tree, and Random Forest.
    #  Each model will be trained and evaluated on the same train/test splits to compare their performance
    model_list = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(), "Random Forest")]

    # first we loop though the different test sizes to create different train/test splits of the data 
    # then we loop through each model to train and evaluate it on those splits. 
    # This allows us to see each model performance based off the 0.3, 0.2, 0.1 tests
    for test_size in test_sizes:
        print(f"\n-------------------------------")
        print(f"Train/Test Split: {int((1 - test_size)*100)} / {int(test_size*100)}") # converts into whole numbers like "Train/Test Split: 70 / 30" for a test size of 0.3). It helps to organize the output and make it clear which results correspond to which split.
        print(f"-------------------------------")
        # 
        X_train, X_test, y_train, y_test = train_test_split(
            X, # the input features for the model
            y, # the target variable (0 for human-generated, 1 for AI-generated)
            test_size=test_size, # the proportion of the dataset to include in the test split (0.3, 0.2, 0.1)
            random_state=42 #
            )

        # for each test size loop through each model and train and evaluate it on the current train/test split
        for model, name in model_list:
            train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run_models()