from crypt import methods
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from flask import Flask,request,jsonify,render_template
from flask import  request, send_from_directory
#from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
sentiment_model = load_model("Sentiment/")

app = Flask(__name__)
#app = Flask(__name__, static_folder="./frontend/build/", static_url_path="")
#CORS(app)

linebreaks        = "<br /><br />"
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_reviews(review):

    review = review.lower()

    review = re.sub(linebreaks," ",review)
    # Replace 3 or more consecutive letters by 2 letter.
    review = re.sub(sequencePattern, seqReplacePattern, review)

    # Replace all emojis.
    review = re.sub(r'<3', '<heart>', review)
    review = re.sub(smileemoji, '<smile>', review)
    review = re.sub(sademoji, '<sadface>', review)
    review = re.sub(neutralemoji, '<neutralface>', review)
    review = re.sub(lolemoji, '<lolface>', review)

    # Remove non-alphanumeric and symbols
    review = re.sub(alphaPattern, ' ', review)

    # # Adding space on either side of '/' to seperate words (After replacing URLS).
    # review = re.sub(r'/', ' / ', review)
    return review
#load tokenizer and pad the sequence using pickle
with open('Tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#make an flask api which accepts a statement and returns the sentiment of the statement using model stored in Sentiment and gets argument using a q parameter in GET request
def get_sentiment(text):
    statement = [text]
    for i in range(len(statement)):
        listr = []
        statement[i] = preprocess_reviews(statement[i])
        for word in statement[i].split():
            if word.lower() not in stop_words:
                listr.append(word)
        statement[i] = " ".join(listr)
    print(statement)
    statement = pad_sequences(tokenizer.texts_to_sequences(statement) , maxlen=1500)
    pred = sentiment_model.predict(statement)
    pred = np.where(pred>=0.5, 1, 0)
    return pred[0][0]
# at / route return a form to enter the statement and render the sentiment



@app.route('/',methods=['GET','POST'])
def sentiment():
    if request.method == 'POST':
        q = request.form['q']
        
        if q is not None and q != '':
            statement = q
            sentiment = get_sentiment(statement)
            if sentiment == 1:
                output = {'sentiment': 'positive'}
            else:
                output = {'sentiment': 'negative'}
        else:
            output = {'sentiment': 'query not provided'}
        return render_template('index.html',output=output,text=q)
    else:
        text = "Text"
        return render_template('index.html',text=text,output="No query")

#@app.route("/", methods=["GET", "POST"])
#@cross_origin()
#def serve():
#    return send_from_directory(app.static_folder, "index.html")
    

if __name__ == '__main__':
    app.run(debug=True)

