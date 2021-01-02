import requests
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
model = load_model("DVC2.h5",compile=True)
# model._make_predict_function() 


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
        # Get the file from post request
		f = request.files['file']

	    # Save the file to .uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
		basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make prediction
		img = image.load_img(file_path, target_size=(64, 64))
		# img = cv2.resize(img,(64,64))
		img = np.reshape(img,[1,64,64,3])
		image = tf.cast(img, tf.float32)
		classes = (model.predict(img) > 0.5).astype("int32")

		# Process your result for human
		# pred_class = preds.argmax(axis=-1)            # Simple argmax
		# pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
		# result = str(pred_class[0][0][1])               # Convert to string
		if classes[0][0] == 1:
			return "DOG"
		else:
			return "CAT"
		# return str(classes)
	return None


if __name__ == '__main__':
    app.run(debug=False)
