# app.py

from flask import Flask, jsonify, render_template, request
from textblob import TextBlob
import pusher
import os
import sys
sys.path.insert(0,'..')

from base_line_DNN import CNN_LSTM as network
from base_line_DNN import ProcessData

app = Flask(__name__)
net = network()

pusher = pusher.Pusher(
    app_id=os.getenv('PUSHER_APP_ID'),
    key=os.getenv('PUSHER_APP_KEY'),
    secret=os.getenv('PUSHER_APP_SECRET'),
    cluster=os.getenv('PUSHER_APP_CLUSTER'),
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
    out = net.predict(model_path='../../model/hi_cdmx_weights.best.hdf5',x=test_data.X)
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
    app.run(port=8893)