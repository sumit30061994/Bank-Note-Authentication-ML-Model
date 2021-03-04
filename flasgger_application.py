# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:28:04 2021

@author: sumit
"""
from flask import Flask, request
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_input = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_input)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict', methods=['Get'])
def predict_note_authentication():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "the predicted value is " + str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_note_auth():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))
    

if __name__ == '__main__':
    app.run()
