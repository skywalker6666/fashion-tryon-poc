from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from utils.modnet_segment import segment_person
from utils.warp_clothes import warp_clothing_onto_person

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

mp_pose = mp.solutions.pose


def extract_pose_landmarks(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        points = {
            'lx': int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
            'ly': int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h),
            'rx': int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
            'ry': int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h),
            'lxw': int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
            'lyw': int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h),
            'rxw': int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
            'ryw': int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h),
        }
        return points


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    user_img = request.files['user_image']
    clothes_img = request.files['clothes_image']

    user_filename = secure_filename(user_img.filename)
    clothes_filename = secure_filename(clothes_img.filename)

    user_path = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
    clothes_path = os.path.join(app.config['UPLOAD_FOLDER'], clothes_filename)
    user_img.save(user_path)
    clothes_img.save(clothes_path)

    # Step 1: 人像去背
    user_rgba_filename = f"rgba_{user_filename}"
    user_rgba_path = os.path.join(app.config['UPLOAD_FOLDER'], user_rgba_filename)
    segment_person(user_path, user_rgba_path)

    # Step 2: 取得人體貼合點
    pose_points = extract_pose_landmarks(user_path)
    if not pose_points:
        return "Failed to detect pose landmarks."
    offset_y = 10
    dst_points = [
        [pose_points['lx'], pose_points['ly'] - offset_y],
        [pose_points['rx'], pose_points['ry'] - offset_y],
        [pose_points['lxw'], pose_points['lyw'] + offset_y],
        [pose_points['rxw'], pose_points['ryw'] + offset_y]
    ]

    clothes_img = cv2.imread(clothes_path, cv2.IMREAD_UNCHANGED)
    h, w = clothes_img.shape[:2]
    src_points = [
        [0.2 * w, 0.1 * h],  # 左肩
        [0.8 * w, 0.1 * h],  # 右肩
        [0.2 * w, 0.9 * h],  # 左下
        [0.8 * w, 0.9 * h],  # 右下
    ]

    # Step 4: 執行 warp 並貼合
    output_filename = f"result_{user_filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    warp_clothing_onto_person(
        clothes_path,
        user_rgba_path,
        output_path,
        src_points,
        dst_points
    )

    return render_template('result.html', result_image=output_filename)


if __name__ == "__main__":
    app.run(debug=True)
