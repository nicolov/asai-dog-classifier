#!/usr/bin/env python

import os

from flask import Flask, request, redirect, send_from_directory

app = Flask(__name__)

# create uploads directory
os.makedirs("uploads", exist_ok=True)

# import tensorflow


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return redirect("/")
    save_path = os.path.join("uploads", uploaded_file.filename)
    uploaded_file.save(save_path)
    return redirect("classify/" + uploaded_file.filename)


@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory("uploads", filename)


@app.route("/classify/<filename>")
def classify(filename):
    model_result = "dog"

    return """
<title>Classification result</title>
<h1>The model says .. {result}</h1>

<img width=400px src="/images/{filename}" />

""".format(
        filename=filename, result=model_result
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
