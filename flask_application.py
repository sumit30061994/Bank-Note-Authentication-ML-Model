# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:28:04 2021

@author: sumit
"""
from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

pickle_input = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_input)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "the predicted value is " + str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_note_auth():
    # use postman
    file = pd.read_csv(request.files.get("input_data"))
    prediction = classifier.predict(file)
    return "The predicted value for the test files is ", str(list(prediction))
    

if __name__ == '__main__':
    app.run()