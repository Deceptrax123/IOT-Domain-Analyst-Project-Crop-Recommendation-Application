from flask import Flask,render_template,request 
import tensorflow as tf 
import keras
import numpy as np 

model=tf.keras.models.load_model("Model/model.keras")
app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']

    arr=np.array([[float(data1),float(data2),float(data3),float(data4),float(data5),float(data6),float(data7)]])
    pred=model.predict(arr)

    data=int(np.where(pred[0]==1)[0])

    return render_template('after.html',data=data)


if __name__=='__main__':
    app.run(debug=True)