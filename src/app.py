import base64
import cv2
import numpy as np

from flask import Flask, request
from flask_cors import CORS
from PIL import Image
from numpy import asarray, expand_dims
from scipy.spatial.distance import cosine

from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os
import pandas as pd

global names
global database_embedding
app = Flask(__name__)
CORS(app)

def extract_face(filename, required_size=(224, 224)):

	pixels = plt.imread(filename)
	detector = MTCNN()
	try: 
		results = detector.detect_faces(pixels)
		x1, y1, width, height = results[0]['box']
	except:
		raise ValueError("No Face Detected")
	
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


def get_embeddings(filenames):
	faces = [extract_face(f) for f in filenames]
	samples = asarray(faces, 'float32')
	samples = preprocess_input(samples, version=2)
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	yhat = model.predict(samples)
	return yhat

def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    idx = image_base64.find('base64,')
    image_base64  = image_base64[idx+7:]

    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2

def get_result(yhat):
	cosine_similarity = [cosine(yhat[0],item) for item in database_embedding]
	database = pd.DataFrame(data = {"name" : names, "cosine" : cosine_similarity})
	database = database.sort_values(by=['cosine'], ignore_index=True)

	if database['cosine'][0] <= 0.5:
		name = os.path.splitext(database['name'][0])[0]
		name = name.replace("_", " ").split(" ")
		name = " ".join( i.capitalize() for i in name)
		return "Welcome " + name
	else:
		return "No Authorized Personnel detected"

def get_face_embedding(pixels):
	pixels = pixels.astype('float32')
	samples = expand_dims(pixels, axis=0)
	samples = preprocess_input(samples, version=2)
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	yhat = model.predict(samples)
	return yhat

# detect person and get name
def detect_person(input_path):
	try:
		pixels = extract_face(input_path)
		plt.imshow(pixels)
		plt.axis("off")
		record_name  = datetime.now().strftime("%m%d%Y%H%M%S")
		path = os.path.join("record", record_name)
		plt.savefig(path)
	except (ValueError): 
		return "No Face Detected"

	yhat = get_face_embedding(pixels)
	return get_result(yhat)

@app.route("/recognition", methods=['POST'])
def recognition():
	input_path = 'tmp.jpg'
	if 'file' not in request.files:
		input = base64_to_cv2(request.json["image"])
		cv2.imwrite(input_path, input)
	else:
		request.files["file"].save(input_path)
	return detect_person(input_path)

names = os.listdir("database")
database_path = [os.path.join("database", image) for image in names]
database_embedding = get_embeddings(database_path)
app.run(debug=True)