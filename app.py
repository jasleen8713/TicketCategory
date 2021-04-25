import flask
import pandas as pd
import tensorflow as tf
import keras
keras.__version__
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from textblob import TextBlob

# instantiate flask 
app = flask.Flask(__name__)

#sentiment analysis
def sentiment_analysis(ticket_text):
  score = TextBlob(ticket_text).sentiment.polarity
  sentiment = ""
  if score < 0:
    sentiment = "Negative"
  elif score == 0:
    sentiment = "Neutral"
  else:
    sentiment = "Positive"
    
  return sentiment

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, echo the msg parameter 
    if (params != None):
        x=params.get("description")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(x)
        x = tokenizer.texts_to_sequences(x)
        maxlen = 150
        x = pad_sequences(x, padding='post', maxlen=maxlen)
       
        ticket_model = load_model('tickettype_model.h5')
        ticket_pred = (ticket_model.predict(x) > 0.5).astype("int32")

        categ_model = load_model('tickettype_model.h5')
        categ_pred = np.argmax(categ_model.predict(x),axis=-1)

        impact_model = load_model('tickettype_model.h5')
        impact_pred = np.argmax(impact_model.predict(x),axis=-1)
            

    # return a reponse in json format 
    data = {'ticket type': ticket_pred, 'category' : categ_pred, 'impact' : impact_pred, 'sentiment' : sentiment}
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')