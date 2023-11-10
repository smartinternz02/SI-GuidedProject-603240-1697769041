import numpy as np
import os
import tensorflow as tf
from PIL import Image
from flask import Flask,render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model('Dogbreed.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/output",methods=['GET','POST'])
def output():
    if request.method=='POST':
        f =request.files['File']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(224,224))
        
        image_array=np.array(img)
        image_array=np.expand_dims(image_array,axis=0)
        
        pred = np.argmax(model.predict(image_array),axis=1)
        print(pred)
        return render_template("output.html",predict=prediction)
    else:
        return render_template("output.html")
    
if __name__=='__main__':
    app.run(debug=False,threaded=False)
