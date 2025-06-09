from PIL import Image
import cv2
import mediapipe as mp
import math


def overlay_clothes_with_pose(user_path, clothes_path, output_path, y_offset=20):
    # Step 1: 偵測人體關鍵點
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    img_cv = cv2.imread(user_path)
    h, w = img_cv.shape[:2]
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        print("❌ 無法偵測到人體姿勢")
        return

    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    lx, ly = int(left_shoulder.x * w), int(left_shoulder.y * h)
    rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)

    center_x = (lx + rx) // 2
    top_y = min(ly, ry) - y_offset
    shoulder_width = abs(rx - lx)

    # Step 2: 疊圖處理
    user_img = Image.open(user_path).convert("RGBA")
    clothes_img = Image.open(clothes_path).convert("RGBA")

    new_width = shoulder_width
    ratio = new_width / clothes_img.width
    new_height = int(clothes_img.height * ratio)
    clothes_resized = clothes_img.resize((new_width, new_height))
    # 取得髖部位置
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    hx = int((left_hip.x + right_hip.x) / 2 * w)
    hy = int((left_hip.y + right_hip.y) / 2 * h)

    shoulder_y = int((ly + ry) / 2)
    clothing_height = hy - shoulder_y

    # 根據 clothing_height 計算比例
    ratio = clothing_height / clothes_img.height
    new_width = int(clothes_img.width * ratio)
    new_height = clothing_height
    clothes_resized = clothes_img.resize((new_width, new_height))
    # 計算貼圖位置（讓衣服中心對齊肩膀中心）
    angle = math.degrees(math.atan2(ry - ly, rx - lx))  # 順時針為正
    rotated = clothes_resized.rotate(-angle, expand=True)
    paste_x = center_x - rotated.width // 2
    paste_y = top_y

    user_img.paste(clothes_resized, (paste_x, paste_y), clothes_resized)
    user_img.save(output_path)
    print(f"✅ 合成完成: {output_path}")
