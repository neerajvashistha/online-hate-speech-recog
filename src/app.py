# app.py

from flask import Flask, jsonify, render_template, request
from textblob import TextBlob
from flask import flash
import pusher
import os
import sys
import pandas as pd
sys.path.insert(0,'..')

from base_line_DNN import CNN_LSTM as network
from base_line_DNN import ProcessData
from langdetect import DetectorFactory
from langdetect import detect
DetectorFactory.seed = 0

app = Flask(__name__)
parameters={'lang':'en',
       'seed':30,
       'epochs':50, 
       'batch_size':30, 
       'optimiser':'sgd', 
       'lr_rate':0.01, 
       'drop_out':0.2, 
       'hidden_size':64,
            'val_split':0.2
      }


net = network(param=parameters)

PUSHER_APP_ID='1058463'
PUSHER_APP_KEY='6214a344fdd8b733dfd9'
PUSHER_APP_SECRET='4bb63530ff3fb7a73432'
PUSHER_APP_CLUSTER='eu'

pusher = pusher.Pusher(
    app_id=PUSHER_APP_ID,
    key=PUSHER_APP_KEY,
    secret=PUSHER_APP_SECRET, 
    cluster=PUSHER_APP_CLUSTER, 
    ssl=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_comment', methods=["POST"])
def add_comment():
    # Extract the request data
    request_data = request.get_json()
    id = request_data.get('id', '')
    username = request_data.get('username', '')
    comment = request_data.get('comment', '')
    socket_id = request_data.get('socket_id', '')
    if detect(comment)=='en':
        print('English')
        test_data = ProcessData(df=list(comment), lang='en', max_seq_len=138)
        out = net.predict(model_path='../../model/en_weights.best.hdf5',x=test_data.X)
        # Get the sentiment of a comment
    #     text = TextBlob(comment)
        print(type(out[0][0]), out[0][0])
        sentiment =  int(out[0][0])
        df = pd.DataFrame([{"text":comment,"class":sentiment,"pedictions":str(out)}])
        df.to_csv('new_train.csv', header=False, mode='a', index=False)
    else:
        print('Hindi')
        test_data = ProcessData(df=list(comment), lang='hi', max_seq_len=105)
        out = net.predict(model_path='../../model/hi_weights.best.hdf5',x=test_data.X)
        # Get the sentiment of a comment
    #     text = TextBlob(comment)
        print(type(out[0][0]), out[0][0])
        sentiment =  int(out[0][0])
        df = pd.DataFrame([{"text":comment,"class":sentiment,"pedictions":str(out)}])
        df.to_csv('new_train.csv', header=False, mode='a', index=False)
    comment_data = {
        "id": id,
        "username": username,
        "comment": comment,
        "sentiment": sentiment,
    }
    
    #  Trigger an event to Pusher
    pusher.trigger(
        "live-comments", 'new-comment', comment_data, socket_id
    )
    
    return jsonify(comment_data)

# run Flask app
if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0', port=8893)