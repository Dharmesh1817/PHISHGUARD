#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import os
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()

sms_model = pickle.load(open("pickle/sms_model.pkl", "rb"))
sms_vectorizer = pickle.load(open("pickle/sms_vectorizer.pkl", "rb"))

email_model = pickle.load(open("pickle/email_model.pkl", "rb"))
email_vectorizer = pickle.load(open("pickle/email_vectorizer.pkl", "rb"))


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


@app.route("/sms", methods=["GET", "POST"])
def sms():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form["sms_text"]
        import re
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        text = message.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
        x = sms_vectorizer.transform([text])
        pred = sms_model.predict(x)[0]
        proba = sms_model.predict_proba(x)[0]
        if pred == 1:
            result = "unsafe"
            confidence = round(proba[1] * 100, 2)
        else:
            result = "safe"
            confidence = round(proba[0] * 100, 2)
        return render_template("sms.html", result=result, confidence=confidence, message=message)
    return render_template("sms.html", result=None, message="")


@app.route("/email", methods=["GET", "POST"])
def email():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form["email_text"]
        import re
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        text = message.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
        x = email_vectorizer.transform([text])
        pred = email_model.predict(x)[0]
        proba = email_model.predict_proba(x)[0]
        if pred == 1:
            result = "unsafe"
            confidence = round(proba[1] * 100, 2)
        else:
            result = "safe"
            confidence = round(proba[0] * 100, 2)
        return render_template("email.html", result=result, confidence=confidence, message=message)
    return render_template("email.html", result=None, message="")


if __name__ == "__main__":
    app.run(debug=True)
