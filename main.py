#!/usr/bin/env python

import os

from flask import Flask, request, redirect, send_from_directory
import tensorflow as tf

from model import get_model
from eval import predict_file_path

app = Flask(__name__)

global graph
global model
model = get_model(weights=None)
model.load_weights("./trained_model.chkpt")
graph = tf.get_default_graph()

# create uploads directory
os.makedirs("uploads", exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return redirect("/")
    save_path = os.path.join("uploads", uploaded_file.filename)
    uploaded_file.save(save_path)
    return redirect("/classify/" + uploaded_file.filename)


@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory("uploads", filename)


@app.route("/classify/<filename>")
def classify(filename):
    image_path = os.path.realpath(os.path.join('uploads', filename))
    global graph
    with graph.as_default():
        result = predict_file_path(model, image_path)

    result_label = 'dog' if result[0][0] > 0.5 else 'cat'

    return """
<title>Classification result</title>
<h1>The model says .. {result}</h1>
<h2>Loaded {filename}</h2>

<p>
    <a href="/">Go back</a>
</p>

<p>
    <img width=400px src="/images/{filename}" />
</p>

""".format(
        filename=filename, result=result_label
    )


@app.route("/")
def home():
    return """
<title>Upload a new image</title>
<h1>Upload a new image</h1>

<form action="/upload" method=post enctype=multipart/form-data>
    <p>
        <input type=file name=file>
    </p>
    <p>
        <input type=submit value=Upload>
    </p>
</form>
"""


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
