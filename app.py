from flask import Flask,render_template,request 
import tensorflow as tf 
import keras
import numpy as np 

model=tf.keras.models.load_model("Model/model.keras")
means=[50.551818181818184,53.36272727272727,48.14909090909091,25.616243851779544,71.48177921778637,6.469480065256364,103.46365541576817]
stds=[36.90894257695227,32.97838509495386,50.636418345000635,5.062597617195944,22.25875105745574,0.7737617731081714,54.945896562329025]

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def home():
    #Get Data
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']

    form_values=[data1,data2,data3,data4,data5,data6,data7]
    processed_values=list()

    #Pre-Process
    for c,i in enumerate(form_values):
        processed_values.append((float(i)-means[c])/stds[c])

    arr=np.array([processed_values])

    #Prediction
    pred=model.predict(arr)

    data=np.argmax(pred[0])

    #Render
    return render_template('predict.html',data=data)


if __name__=='__main__':
    app.run(debug=True)