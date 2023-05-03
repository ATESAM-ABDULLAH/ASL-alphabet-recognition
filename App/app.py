#Import necessary libraries
from flask import Flask, render_template, Response
from camera import *

#Initialize the Flask app
app = Flask(__name__)

#Default app route
@app.route('/')
def video_feed():
   return 


if __name__ == "__main__":
    app.run(debug=True)