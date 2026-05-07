#model.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler



# the extracted dataa features that we will use for training the models
FEATURES = [
    'word_count',
    'sentence_count',
    'avg_sentence_length',
    'vocab_diversity',
    'punctuation_freq',
    'avg_word_length',
    'stopword_ratio',
    'paragraph_count'
]

# ------ LOAD DATA ------ 
# Load data from SQLite database
def load_data():
    conn = sqlite3.connect('database/ai_human.db')
    df = pd.read_sql("SELECT * FROM extracted_texts", conn)
    conn.close()
    return df

# ------ TRAIN AND EVALUATE MODEL------ 
# Train and evaluate the model
# This function takes a model, its name, training and testing data, fits the model, makes predictions, and prints the accuracy and classification report.
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    # LEARN PATTERNS
    # fit: This method is used to train the model on the training data (X_train and y_train). It adjusts the model's parameters to minimize the error between the predicted and actual labels.
    # X_train is the feature data used for training the model and y_train is the corresponding labels (0 for human-generated & 1 for AI-generated)
    model.fit(X_train, y_train)

    # APPLY LEARNED PATTERNS
    # model.predict: This method is used to make predictions on the test data (X_test) using the trained model.
    predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, predict)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    print(classification_report(y_test, predict))

    return accuracy

#PART 1
# ========= TEST DIFFERENT TRAIN/TEST SPLITS =========
def run_baseline_models(X, y):
    print("\n" + "=" * 60)
    print("PART 1: MULTIPLE TRAIN/TEST SPLITS")
    print("=" * 60)

    # Different test sizes (30%, 20%, 10%)
    # 0.3 = 30% for testing and 70% for training
    # 0.2 = 20% for testing and 80% for training
    # 0.1 = 10% for testing and 90% for training.
    test_sizes = [0.3, 0.2, 0.1]

    # We will evaluate three different models: Logistic Regression, Decision Tree, and Random Forest.
    # Each model will be trained and evaluated on the same train/test splits to compare their performance.
    model_list = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(random_state=42), "Random Forest")
    ]

    # first we loop through the different test sizes to create different train/test splits of the data
    # then we loop through each model to train and evaluate it on those splits.
    # This allows us to see each model performance based off the 0.3, 0.2, 0.1 tests.
    for test_size in test_sizes:
        print(f"\n-------------------------------")
        print(f"Train/Test Split: {int((1 - test_size) * 100)} / {int(test_size * 100)}")
        print(f"-------------------------------")

        X_train, X_test, y_train, y_test = train_test_split(
            X,  # the input features for the model
            y,  # the target variable (0 for human-generated, 1 for AI-generated)
            test_size=test_size,  # the proportion of the dataset to include in the test split
            random_state=42  # keeps the split the same each time the code runs
        )

        # for each test size, loop through each model and train/evaluate it on the current train/test split
        for model, name in model_list:
            train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

# PART 2
# =========CROSS-VALIDATION =========
def run_cross_validation(X, y):
    print("\n" + "=" * 50)
    print("PART 2: K-FOLD CROSS-VALIDATION")
    print("=" * 50) 

    # Cross-validation tests the model on multiple different splits of the dataset
    # This gives a more reliable idea of performance than only using one train/test split.
    # KFold splits the data into 5 different folds, shuffling the data before splitting to ensure randomness, and using a fixed random state for reproducibility.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    model_list = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(random_state=42), "Random Forest")
    ]

    for model, name in model_list:
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        print(f"\n--- {name} ---")
        print(f"Fold Scores: {[round(score, 4) for score in scores]}")
        print(f"Average Accuracy: {scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}")

# PART 3
# ========= PART 3: HYPERPARAMETER TUNING =========
def run_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("PART 3: HYPERPARAMETER TUNING")
    print("=" * 50)

    # HYPERPARAMETER TUNING is the process of finding the best settings for a machine learning model to improve its performance.
    # GRIDSEARCHCV is a method that tests all combinations of specified hyperparameters to find the best one based on cross-validation performance.
    # Decision Tree tuning 

    print("\n--DECISION TREE:--")
    # The hyperparameters we will tune for the Decision Tree model include
    decision_tree_params = {
        
        'max_depth': [5, 10, 20, None], # max_depth controls how deep the tree can grow. A deeper tree can capture more complex patterns but may overfit, while a shallower tree may underfit.
        'min_samples_split': [2, 5, 10], # min_samples_split controls the minimum number of samples required to split an internal node. Higher values prevent overfitting.  
        'criterion': ['gini', 'entropy'] # criterion determines the function used to measure the quality of a split. 'gini' is the Gini impurity, while 'entropy' is the information gain. Testing both can help find which one works better for our dataset.
    }

    # GridSearchCV will test all combinations of the specified hyperparameters for the Decision Tree model using 3-fold cross-validation and evaluate performance based on accuracy.
    decision_tree_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42), # the model we want to tune
        decision_tree_params, # the hyperparameters we want to test
        cv=3, # the number of folds for cross-validation (3 means the data will be split into 3 parts, and the model will be trained on 2 parts and tested on the remaining part, repeated 3 times)
        scoring='accuracy' 
    )

    # .fit() runs the grid search process, where it trains the Decision Tree model on different combinations of hyperparameters and evaluates their performance using cross-validation to find the best set of hyperparameters based on accuracy.
    decision_tree_grid.fit(X_train, y_train) 

    # GET BEST COMBO OF PARAMETERS
    # EX output: {'max_depth': 10, 'min_samples_split': 5, 'criterion': 'gini'}
    # these tell us that the best Decision Tree model was achieved with a max_depth of 10, min_samples_split of 5, and using the 'gini' criterion for splitting.
    best_params = decision_tree_grid.best_params_
    print("Best Decision Tree Parameters:", best_params)
    print(f"Best Decision Tree CV Accuracy: {decision_tree_grid.best_score_:.4f}")

    # after best hyperparamters found we can use best estimator
    # .best_estimator_.predict() - choose best model for predictions 
    # predicitions is basically saying based on these features, do you think this text is human or AI generated? (0 or 1)
    decision_tree_predictions =  decision_tree_grid.best_estimator_.predict(X_test)

    # accuracy_score compares the model’s predictions to the true labels and returns the % of correct predictions 
    # true labels are the actual values of y_test (0 for human-generated, 1 for AI-generated) and predictions are the model's guesses based on the features in X_test.
    #  The accuracy score tells us how well the model is performing 
    print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, decision_tree_predictions):.4f}")

    # classification_report gives:
    # ** precision (how accurate predictions are), 
    # ** recall (how well the model finds all true cases),
    # ** F1-score (balance of both), for each class (Human vs AI).
    print(classification_report(y_test, decision_tree_predictions))


    # ****RANDOM FOREST TUNING******
    print("\n--RANDOM FOREST:--")
    forest_params = {
        'n_estimators': [50, 100], # n_estimators is the number of trees in the forest. More trees can improve performance but also increase training time
        'max_depth': [10, 20, None], # max_depth controls how deep each tree can grow. A deeper tree can capture more complex patterns but may overfit, while a shallower tree may underfit
        'max_features': ['sqrt', 'log2'] # max_features determines the number of features to consider when looking for the best split. 'sqrt' uses the square root of the total features, while 'log2' uses the logarithm base 2 of the total features
    }

    # test all combinations of specified hyperparameters for the Random Forest model 
    # using 3-fold cross-validation 
    # evaluate performance based on accuracy.
    random_forest_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        forest_params,
        cv=3,
        scoring='accuracy'
    )

    # .fit() runs the grid search process
    # trains the Random Forest model on different combinations of hyperparameters evaluates their performance using cross-validation 
    # IN ORDER TO = find the best set of hyperparameters based on accuracy.
    random_forest_grid.fit(X_train, y_train)

    print("Best Random Forest Parameters:", random_forest_grid.best_params_) # Print BEST COMBO OF PARAMETERS for Random Forest model
    print(f"Best Random Forest CV Accuracy: {random_forest_grid.best_score_:.4f}") # Print the best cross-validation accuracy achieved with the best hyperparameters 

    # test the best Random Forest model on the x_test to see how it performs on unseen data
    # x_test is the feature data for testing and y_test is the corresponding true labels (0 for human-generated, 1 for AI-generated)
    best_random_forest = random_forest_grid.best_estimator_
    random_forest_predictions = best_random_forest.predict(X_test)

    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, random_forest_predictions):.4f}") # Print the accuracy of the best Random Forest model on the test set (how it wokrs on unseen data)
    print(classification_report(y_test, random_forest_predictions)) # Print the classification report for the best Random Forest model on the test set, which includes precision, recall, and F1-score for each class (Human vs AI)

    return best_random_forest 

# PART 4
# This function analyzes the feature importance of the best Random Forest model to understand which features were most influential in distinguishing between human and AI generated texts
# we only use random forest for feature importance because it has a built-in attribute called feature_importances_ that provides a score for each feature based on how much it contributed to the model's predictions across all the trees in the forest
# also its more stable since it averages importance across many trees, while decision tree importance can be more biased towards features that appear near the top of the tree.
# ============= FEATURE IMPORTANCE ================
def run_feature_importance(best_model):
    print("\n" + "=" * 50)
    print("PART 4: FEATURE IMPORTANCE")
    print("=" * 50)

    # create a dataframe to display the importance of each feature based on the best Random Forest model's feature_importances_ attribute
    # layout of importance_df:
    # Feature           | Importance = a score that indicates how much each feature contributed to the model's predictions (higher values mean more importance)
    # word_count       | 0.25
    # sentence_count   | 0.15
    # ETC......
    importance_df = pd.DataFrame({
        'Feature': FEATURES, # list of feature names that we used for training the model (word_count, sentence_count, etc.)
        'Importance': best_model.feature_importances_ # the importance of each feature as determined by the best Random Forest model, which indicates how much each feature contributed to the model's predictions (higher values mean more importance)
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

from dataextract import extract_data  
#------ CUSTOM USER PREDICTION FUNCTION--------
def predict_text(z_model, scaler):
    print("\n--- PREDICT HUMAN vs AI GENERATED TEXT ---")

    while True:

        user_input = input("Enter text (type 'quit' to quit): ")

        if user_input.lower() == 'quit':
            print("Exiting...")
            break

        features = extract_data(user_input) # use the extract_data function to extract the same features from the user input text that we used for training the model (word_count, sentence_count, etc.)
        df = pd.DataFrame([features])[FEATURES] # create a dataframe with the extracted features and ensure it has the same column order as the FEATURES list

        # scale the user input features using the same scaler that was used to scale the training data (important to ensure the model can make accurate predictions based on the same feature scales)
        # scaling is important because the model was trained on scaled features, so we need to apply the same scaling to the user input features to ensure they are on the same scale for accurate predictions.
        # scaling basically standardizes the feature values to have a mean of 0 and a standard deviation of 1
        #we use col = feature because we recieved warning ab col names not matching
        scaledUserData = pd.DataFrame(scaler.transform(df), columns=FEATURES) 
        predict = z_model.predict(scaledUserData) # use the trained model (z_model) to predict whether the user input text is human-generated (0) or AI-generated (1) based on the extracted and scaled features

       
        if predict[0] == 1:  
            print("Prediction: AI-generated")
        else:
            print("Prediction: Human-written")

        print("DF:", df)
        print("Scaled:", scaledUserData)
        print("Prediction:", predict)
        print("Proba:", z_model.predict_proba(scaledUserData))

# ----------MAIN FUNCTION ----------
def run_models():
    df = load_data()

    X = df[FEATURES] # the input features for the model (word_count, sentence_count, etc.) that we extracted from the text and will use to train the model
    y = df['generated'] # the target variable (0 for human-generated, 1 for AI-generated) that we want the model to learn to predict based on the input features in X

    # scaling standardizes the feature values to have a mean of 0 and a standard deviation of 1
    # which ensures that all features are on the same scale and preventing features with larger ranges from effecting the model learning process
    scale = StandardScaler()
    X_scaled = pd.DataFrame(scale.fit_transform(X), columns=FEATURES) #here we fit the scaler to the training data (X) and then transform it to create a new dataframe (X_scaled) with the same column names as FEATURES, but with all feature values scaled to have a mean of 0 and a standard deviation of 1.

    # this part splits the scaled features (X_scaled) and target variable (y) into training and testing sets
    # 80% of the data is used for training the model and 20% is used for evaluating (standard)
    #  The random_state parameter ensures that the split is reproducible each time the code is run
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    run_baseline_models(X_scaled, y) # this function runs the baseline models (Logistic Regression, Decision Tree, Random Forest) on different train/test splits of the data (30%, 20%, 10%) to evaluate their performance based on accuracy and classification report for each split.
    run_cross_validation(X_scaled, y) # this function runs K-Fold Cross-Validation (with 5 folds) for each of the baseline models to evaluate their performance across multiple different splits of the dataset, providing fold scores, average accuracy, and standard deviation for each model.

    # finds best hyperparameters for Decision Tree and Random Forest models using GridSearchCV,
    # returns the best Random Forest model for further analysis of feature importance and user input prediction
    best_model = run_hyperparameter_tuning(X_train, X_test, y_train, y_test)
    run_feature_importance(best_model)

    predict_text(best_model, scale)

    print("\nmodels ALL ran successfully!")

if __name__ == "__main__":
    run_models()