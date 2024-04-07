from flask import Flask,render_template,request 
import tensorflow as tf 
import keras
import numpy as np 

model=tf.keras.models.load_model("Model/model.keras")
app=Flask(__name__)

@app.route('/')
def man():
    return render_template()