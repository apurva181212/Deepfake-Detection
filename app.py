from flask import Flask, render_template, request

import os
import tensorflow as tf
from utils import extract_frames
import numpy as np

model = tf.keras.models.load_model(
    "deepfake_model.h5",
    compile=False
)


import tensorflow as tf

preprocess_input = tf.keras.applications.resnet50.preprocess_input


# this change is only for account B


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get uploaded video
        video = request.files.get("video", None)
        if video is None or video.filename == "":
            return render_template("index.html", result={"error": "No video uploaded."})

        # Save the uploaded file temporarily
        upload_path = "uploaded.mp4"
        video.save(upload_path)

        # Extract frames from the uploaded video
        frames = extract_frames(upload_path)
        if len(frames) == 0:
            return render_template("index.html", result={"error": "No frames extracted from video."})

        # Resize and preprocess frames (normalize + preprocess_input)
        frames = np.array(frames).astype("float32")
        frames = preprocess_input(frames)

        # Make predictions for each frame
        preds = model.predict(frames, verbose=0)
        avg_pred = float(np.mean(preds))  # average over frames

        # Compute confidence and label
        confidence = round(avg_pred * 100, 2)
        label = "FAKE" if confidence >= 50 else "REAL"

        # Adjust for more human-readable display
        if label == "REAL":
            confidence = round(100 - confidence, 2)

        # Delete uploaded video (optional)
        try:
            os.remove(upload_path)
        except Exception:
            pass

        return render_template("index.html", result={
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return render_template("index.html", result={"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
