import sys
import nltk
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

def load_data(database_filepath):
    
    """
      function:
      load message and categories data from the database and return the input, output and category names for model prep.

      INPUT:
      database_filepath - the path of database where data is stored

      OUTPUT:
      X - Messages column
      Y - Category columns encoded
      Y.columns - Category names
      """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages_categories', engine) 
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y,Y.columns


def tokenize(text):
    
    """
      function:
      Inputs a message, converts to lowercase, tokenizes it, removes stopwords and outputs the lemmatized words in list 

      INPUT:
      text - individual messages

      OUTPUT:
      lemmatized_words - tokenized, stopwords removed, lemmatized words in the message as a list
      """
    #removing all special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    #tokenize the words,remove stopwords, and convert all characters to lower case
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    stop_words = stopwords.words("english")
    words = [word for word in words if word not in stop_words]
    #lemmatize the words 
    lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]
    
    return lemmatized_words


def build_model():
    
    """
    function:
    Model is built in which data can be trained to predict the message category.
    """

    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
        ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    pred_model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return pred_model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    function:
    This function is used to evaluate the predictions of the classification model built.  

    INPUT:
    model - the classification model built
    X_test - The testing set of messages which need to be evaluated
    Y_test - Actual output categories for the testing set
    category_names - list of all category names
    """
    
    prediction = model.predict(X_test)
    for num, column_name in enumerate(category_names):
        print(f"******************{column_name}******************")
        print(classification_report(Y_test[column_name], prediction[:, num]))
    accuracy = (prediction == Y_test.values).mean()
    print(f'Overall accuracy - {accuracy}')

def save_model(model, model_filepath):
    """
    function:
    Saves the model in a pickle file.
    
    INPUT: 
    
    model - The model built
    model_filepath - path where pickle file is stored
        
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)
    
    

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
    
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()