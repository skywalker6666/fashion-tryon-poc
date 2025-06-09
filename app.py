from flask import Flask, render_template, request, send_from_directory
import os
from utils.overlay import overlay_clothes

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    user_img = request.files['user_image']
    clothes_img = request.files['clothes_image']

    user_path = os.path.join(UPLOAD_FOLDER, user_img.filename)
    clothes_path = os.path.join(UPLOAD_FOLDER, clothes_img.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f'result_{user_img.filename}')

    user_img.save(user_path)
    clothes_img.save(clothes_path)

    overlay_clothes(user_path, clothes_path, output_path)

    return send_from_directory(OUTPUT_FOLDER, os.path.basename(output_path))


if __name__ == '__main__':
    app.run(debug=True)
