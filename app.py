import torch
import importlib
import helpers
importlib.reload(helpers)
from flask import Flask, render_template, request, redirect
from pathlib import Path
from helpers import run_prediction

app = Flask(__name__)
uploads_folder = 'static/uploads'
Path(uploads_folder).mkdir(exist_ok=True, parents=True)
app.config['UPLOAD_FOLDER'] = uploads_folder

@app.route('/', methods= ['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = Path(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predicted_class, predicted_idx = run_prediction(img_path=filepath)
            return render_template('index.html', filepath=filepath, predicted_class=predicted_class, predicted_idx=predicted_idx)
    return render_template('index.html', prediction=None, filepath=None, predicted_class=predicted_class, predicted_idx=predicted_idx)


if __name__ == '__main__': 
    app.run(debug=True)