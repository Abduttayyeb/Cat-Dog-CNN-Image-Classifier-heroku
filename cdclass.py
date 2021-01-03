#!/usr/bin/env python
# coding: utf-8

# In[1]:


# if there is keras_scratch_graph error
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')


# In[2]:


from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from flask import send_from_directory



# In[3]:


ALLOWED_EXTENSION  =set(['txt', 'pdf', 'png','jpg','jpeg','gif'])
IMAGE_HEIGHT =64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3


# In[4]:


def allowed_file(filename):
    return '.' in filename and      filename.rsplit('.',1)[1] in ALLOWED_EXTENSION


# In[5]:


app = Flask(__name__)
model = load_model('DVC2.h5',compile=True)

@app.route('/')
def index():
    return render_template('ImageML.html')


# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                                'favicon.ico', mimetype='image/png')



@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select an image')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("***"+filename)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img  = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.reshape(img,[1,64,64,3])
        image = tf.cast(img, tf.float32)
        # x  = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x  = preprocess_input(x)
        cl = model.predict(image)
        # lst =  decode_predictions(pred, top=3)
        
        # items = []
        # for item in lst[0]:
        #     items.append({'name': item[1], 'prob': float(item[2])})
        
        response = (cl>0.5).astype("int32")
        if response[0][0]==1:
            return render_template('ImageML.html', prediction = 'DOG')
        else:
            return render_template('ImageML.html', prediction = 'CAT')


    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
            


# Type 'localhost:5000/index' after this whole script is run i.e. flask server is on. static files and templates are already linked with flask server. After running this script, just enter the localhost address in browser and run classify the images
