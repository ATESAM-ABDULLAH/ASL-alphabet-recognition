#Import necessary libraries
from flask import Flask, render_template, Response
import cv2

#Initialize the Flask app
app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)