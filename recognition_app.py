from flask import Flask, request, jsonify
import requests
import cv2
from deepface import DeepFace
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/verify',methods= ['GET'])
def health_check():
    return jsonify({"status": "alive"}), 200

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
        
 def build_provider_vector(provider, service_count, subservice_count):
    vector = []

    # One-hot service
    service_vec = [0] * service_count
    if provider["service_id"] is not None:
        service_vec[provider["service_id"]] = 1
    vector.extend(service_vec)

    # Multi-hot subservices
    sub_vec = [0] * subservice_count
    for sid in provider["subservices"]:
        if sid < subservice_count:
            sub_vec[sid] = 1
    vector.extend(sub_vec)

    # Numeric features
    vector.append(provider["avg_rating"] / 5.0)
    vector.append(min(provider["jobs_done"], 100) / 100.0)

    return np.array(vector, dtype=float)


def build_client_profile(history, provider_vectors):
    if len(history["provider_ids"]) == 0:
        return np.zeros(len(next(iter(provider_vectors.values()))))

    profile_vectors = []
    for pid in history["provider_ids"]:
        if pid in provider_vectors:
            profile_vectors.append(provider_vectors[pid])

    if not profile_vectors:
        return np.zeros(len(next(iter(provider_vectors.values()))))

    return np.mean(np.array(profile_vectors), axis=0)


@app.post("/recommend/providers")
def recommend():
    data = request.json

    providers = data["providers"]
    history = data["client_history"]
    service_count = data["meta"]["service_count"]
    subservice_count = data["meta"]["subservice_count"]

    provider_vectors = {}
    vectors_list = []

    # Build provider vectors
    for provider in providers:
        vec = build_provider_vector(provider, service_count, subservice_count)
        provider_vectors[provider["id"]] = vec
        vectors_list.append(vec)

    # Build client profile
    client_profile = build_client_profile(history, provider_vectors).reshape(1, -1)

    # Compute similarities
    similarities = []
    for p in providers:
        pid = p["id"]
        sim = cosine_similarity(client_profile, provider_vectors[pid].reshape(1, -1))[0][0]
        similarities.append((pid, sim))

    # Sort descending
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)

    return jsonify({
        "recommended_provider_ids": [pid for pid, _ in ranked]
    })


app.run(host="0.0.0.0",port=5000)

    


