from flask import Flask, request
import base64
import cv2
import numpy as np
from deepface import DeepFace
from deepface.basemodels import Facenet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    idx = image_base64.find('base64,')
    image_base64  = image_base64[idx+7:]

    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2

def detect_person(input_path):
    model = Facenet.loadModel()
    df = DeepFace.find(img_path = input_path, db_path = "database", 
		model_name="Facenet", model = model, distance_metric= 'cosine', detector_backend='ssd')
    df = df.sort_values(by=['Facenet_cosine'])
    if len(df) > 0: 
        if df.iloc[0].Facenet_cosine <= 0.5:
            name = df.iloc[0].identity.split('\\')[1].split('/')
            return name[0]
        else :
            return "None"
    else:
        return "None"

@app.route("/recognition", methods=['POST'])
def recognition():
    if request.method == 'POST': 
        input_path = 'tmp.jpg'
        input = base64_to_cv2(request.json["image"])
        cv2.imwrite(input_path, input)
        result = detect_person(input_path)
        print(result)
        return {
            'response' : result
        }

if __name__ == '__main__':
    app.run(debug=True)
