import os
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.metrics import accuracy_score as acc
from classdef import Pipeline
import dill as pickle
import numpy as np

app = Flask(__name__)

fname = 'churn_model.pk'
with open('./'+fname,'rb') as f:
    pipe = pickle.load(f)

@app.route('/predict', methods=['GET','POST'])
def predict():

    test_json = request.get_json()
    df= pd.read_json(test_json)
    df, dump = pipe.cleaning(df)
    probs = pipe.probs(df)

    if probs[0,0]<0.1:
        pred='High churn risk.'
    else:
        pred='Low churn risk.'
    return jsonify(pred)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=80)
