from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import numpy as np
import joblib
import pickle

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def hello():
#     df = pd.DataFrame()
#     """Return a friendly HTTP greeting."""
#     return 'Hello World!'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = [ 'Credit_History', 'Property_Area', 'Married', 'LoanAmount'])                         

    output = model.predict(data_unseen)
    
    if output == 1:
        label = 'approved'
    else:
        label = 'not approved'

    return render_template('index.html', prediction_text='Loan Status is {}'.format(label))


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]