# model.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    conn = sqlite3.connect('/Users/alinan/Documents/human-ai-detector/database/ai_human.db')
    df = pd.read_sql("SELECT * FROM texts_features", conn)
    conn.close()
    return df

def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

def run_models():
    df = load_data()

    features = ['word_count', 'sentence_count', 'avg_sentence_length',
                 'vocab_diversity', 'punctuation_freq']

    X = df[features]
    y = df['generated']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(), "Random Forest")
    ]

    for model, name in models:
        train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run_models()