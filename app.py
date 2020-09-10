import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

app = Flask(__name__)
bow_transformer = joblib.load('bow_transformer.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form.get("text")
    nopunct=[char for char in text if char not in string.punctuation]
    nopunct=''.join(nopunct)
    lemmatiser = WordNetLemmatizer()
    refined_text = ''
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        refined_text=refined_text+b+' '

    text_bow = bow_transformer.transform([refined_text])
    tf_idf_text = tfidf_transformer.transform(text_bow)
    prediction = model.predict(tf_idf_text)
    if prediction[0] == 'EAP':
        output = 'Edgar Allan Poe'
    elif prediction[0] == 'HPL':
        output = 'HP Lovecraft'
    else:
        output = 'Mary Wollstonecraft Shelley'

    return render_template('index.html', prediction_text='The text may belong to {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)