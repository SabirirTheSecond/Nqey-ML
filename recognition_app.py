from flask import Flask, request, jsonify
import requests
import cv2
from deepface import DeepFace
import numpy as np
from io import BytesIO
from PIL import Image
app = Flask(__name__)



def load_and_resize_image_from_url(url, target_size=(300, 300)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize(target_size)  # Resize to smaller dimensions
    return np.array(img)

@app.route('/verify',methods=['POST'])
def verify():

    try:
       data = request.get_json(force=True)
       print("[INFO] Received payload:", data)
       img1_url = data.get("img1_path")
       img2_url = data.get("img2_path")

       img1 = load_and_resize_image_from_url(img1_url)
       img2 = load_and_resize_image_from_url(img2_url)


       print("[INFO] Running DeepFace verification...")
       result = DeepFace.verify(img1_path=img1, img2_path=img2,model_name="SFace", enforce_detection=False)

       print("[INFO] Verification result:", result)
        

       return jsonify({
            "verified": result.get("verified", False),
            "distance": result.get("distance")
        }), 200

    
    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500
app.run(port=5001)
    