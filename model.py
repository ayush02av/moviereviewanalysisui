import re
import numpy as np
import pandas as pd
import pickle as pkl
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# required assets
vectorizer = Word2Vec.load('assets/vectorizer.model')
scaler = pkl.load(open('assets/scaler.pkl', 'rb'))
model = pkl.load(open('assets/model.pkl', 'rb'))
stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

# helper functions
def sentence_average(keys):
    vector_sum = np.array([0.0] * 100)
    
    for key in keys:
        vector_sum += np.array(vectorizer.wv.get_vector(key).tolist())
    
    return vector_sum / len(keys)

# main prediction function
def predict(review: str) -> str:
    # cleaning
    review = re.sub("<br />", "", review)
    review = re.sub("[^a-zA-Z0-9 ]", "", review)

    # stemming
    review = [
        stemmer.stem(word)
        for word in word_tokenize(review)
        if word not in stopwords_english and stemmer.stem(word) in vectorizer.wv.index_to_key
    ]

    review = pd.DataFrame.from_dict({
        'review': [review]
    })

    # vectorizing
    review = review.apply(
        lambda row: sentence_average(row[0]),
        axis = 1,
        result_type = "expand"
    )
    review.columns = [str(i) for i in range(100)]

    # scaling
    review = pd.DataFrame(
        scaler.transform(review),
        columns = review.columns
    )

    # predicting
    prediction = model.predict(review)
    prediction = (prediction > 0.5)

    return "Positive" if prediction[0][0] == True else "Negative"