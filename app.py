from flask import Flask, render_template, request
from pathlib import Path

uploads_folder = 'static/uploads'
Path(uploads_folder).mkdir(exist_ok=True, parents=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = uploads_folder
# req_f = request.files['file']
app_conf = app.config
@app.route('/')
def send_req():
    names = 'Faiz'
    return render_template('index.html', name=names, app_conf=app_conf)


if __name__ == '__main__': 
    app.run(debug=True)