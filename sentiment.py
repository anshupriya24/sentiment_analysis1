# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 21:45:47 2023

@author: santosh Turamari
"""

from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

app = Flask(__name__)
with open('sentiment_model.pkl', 'rb') as model_file:
    tfidf_vectorizer, clf = pickle.load(model_file)

@app.route('/')
def welcome():
    return "Sentiment Analysis"
    
@app.route('/predict_sentiment', methods=["POST"])
def predict_sentiment():
    try:
        input_text = request.json['text']
        cleaned_new_text = preprocess_text(input_text)
        input_vector = tfidf_vectorizer.transform([cleaned_new_text])
        predicted_sentiment = clf.predict(input_vector)[0]
        response = {'sentiment': predicted_sentiment}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})
    

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return " ".join(filtered_words)



if __name__=='__main__':
    app.run()