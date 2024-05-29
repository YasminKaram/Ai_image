from flask import Flask, request, jsonify,Response,Blueprint
import cv2
import numpy as np
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import base64 
from  Bi_image import bicep_image
from  jump_image import jump_image 
from  squat_Image import squat_image 
#from  YogaF import yoga_image 


app = Flask(__name__)
app.register_blueprint(bicep_image)
app.register_blueprint(jump_image)
app.register_blueprint(squat_image)
#app.register_blueprint(yoga_image)
@app.route('/',methods=['POST'])
def hello():
    return "FITIFY APP"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')