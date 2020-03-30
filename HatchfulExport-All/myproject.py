from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Read dataset by pandas use function read_csv
    messages = pd.read_csv("Data_set/spam.csv", encoding='utf-8', engine='python')
    # Drop the extra columns and create columns
    messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    # Create 2 name column (category, text)
    messages.columns = ["category", "text"]
    # Take dataset category
    messages['category'] = messages['category'].map({'ham': 0, 'spam': 1})
    # Take dataset text
    category = messages['category']
    text = messages['text']
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    # Fit the Data
    text = cv.fit_transform(text)
    # import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(text, category, test_size=0.33, random_state=42)
    # import MultinomialNB - Naive Bayes Classifier
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)
    # Build model with train - Alternative Usage of Saved Model
    # joblib.dump(clf, 'Data_set/NB_spam_model.pkl')

    NB_spam_model = open('Data_set/NB_spam_model.pkl', 'rb')
    clf = joblib.load(NB_spam_model)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
