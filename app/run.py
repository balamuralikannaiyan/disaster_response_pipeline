import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table('disaster_messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #getting the 36 category names and number of messages in each category
    category_names = list(df.columns[4:])
    category_count = list(df[category_names].astype(int).sum(axis = 0))
    
    #getting the category names and number of messages sorted in a dataframe
    category_count_df = pd.DataFrame({'category_names' : category_names,
                                'category_count' : category_count}, 
                                columns=['category_names','category_count'])
    
    category_count_df.sort_values('category_count', ascending = False, inplace = True)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #Have added plots for categories and their respective counts, and for the top ones among them.
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Categorywise count',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    
                    'tickangle': -45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_count_df['category_names'][:10],
                    y=category_count_df['category_count'][:10]
                )
            ],

            'layout': {
                'title': 'Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
    