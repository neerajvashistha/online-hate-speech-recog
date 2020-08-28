# app.py

from flask import Flask, jsonify, render_template, request
from textblob import TextBlob
from flask import flash
import pusher
import os
import sys
sys.path.insert(0,'..')

from base_line_DNN import CNN_LSTM as network
from base_line_DNN import ProcessData

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
# Zelix, 
# text = ['@AvijitEmmi Bahenchod .... experienced lagte ho bhai bhot face with tears of joy']
# test_data = ProcessData(df=text, lang='en', max_seq_len=48)
net = network(param=parameters)
# out = net.predict(model_path='../../model/23-Aug-2020_18_08_51_hi_cdmx_weights.best.hdf5',x=test_data.X)
# print(out)


PUSHER_APP_ID='1058463'
PUSHER_APP_KEY='6214a344fdd8b733dfd9'
PUSHER_APP_SECRET='4bb63530ff3fb7a73432'
PUSHER_APP_CLUSTER='eu'

pusher = pusher.Pusher(
    app_id=PUSHER_APP_ID, #os.getenv('PUSHER_APP_ID'),
    key=PUSHER_APP_KEY, #os.getenv('PUSHER_APP_KEY'),
    secret=PUSHER_APP_SECRET, #os.getenv('PUSHER_APP_SECRET'),
    cluster=PUSHER_APP_CLUSTER, #os.getenv('PUSHER_APP_CLUSTER'),
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
    
    test_data = ProcessData(df=list(comment), lang='en', max_seq_len=48)
    out = net.predict(model_path='../../model/23-Aug-2020_18_08_51_hi_cdmx_weights.best.hdf5',x=test_data.X)
    # Get the sentiment of a comment
#     text = TextBlob(comment)
    print(type(out[0][0]), out[0][0])
    sentiment =  int(out[0][0])
        
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