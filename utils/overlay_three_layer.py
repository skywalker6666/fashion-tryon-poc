from PIL import Image
import cv2
import mediapipe as mp
import math


def overlay_three_layer(user_rgba_path, clothes_path, output_path, bg_color="white"):
    # === 讀取人物透明圖層 ===
    person = Image.open(user_rgba_path).convert("RGBA")
    w, h = person.size

    # === 讀取原始 RGB 圖，用於關鍵點偵測 ===
    img_cv = cv2.imread(user_rgba_path, cv2.IMREAD_UNCHANGED)
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)

    # === 使用 MediaPipe 偵測肩膀位置 ===
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(img_cv_rgb)

    if not results.pose_landmarks:
        print("⚠️ 無法偵測姿勢")
        return

    landmarks = results.pose_landmarks.landmark
    lx = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
    ly = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
    rx = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
    ry = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_y = int((ly + ry) / 2)
    hip_y = int((lhip.y + rhip.y) / 2 * h)
    center_x = (lx + rx) // 2
    shoulder_width = abs(rx - lx)
    clothing_height = hip_y - shoulder_y
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # === 處理衣服圖 ===
    clothes_img = Image.open(clothes_path).convert("RGBA")
    # 以人物肩寬對齊衣服圖片的寬度，等比例縮放
    target_width = shoulder_width * 1.6   # 你可微調倍率
    aspect_ratio = clothes_img.height / clothes_img.width
    target_height = int(target_width * aspect_ratio)
    resized = clothes_img.resize((int(target_width), target_height))
    # rotated = resized.rotate(-angle, expand=True)   ← ❌ 臨時停用
    rotated = resized.copy()                          # ✅ 先保留原始方向
    # === 建立白底背景圖 ===
    bg = Image.new("RGBA", (w, h), bg_color)
    x = center_x - rotated.width // 2
    y = shoulder_y - int(rotated.height * 0.2)
    # === 疊圖順序：背景 → 衣服 → 人物 ===
    # 強制貼圖在中央，作為 debug 用
    bg.paste(person, (0, 0), person)
    bg.paste(rotated, (x, y), rotated)     # 衣服貼在上層 ←✅ 你要的效果

    # bg.paste(rotated, (w//2 - rotated.width//2, h//2 - rotated.height//2), rotated)
    # bg.paste(rotated, (center_x - rotated.width // 2, shoulder_y - 20), rotated)

    bg.save(output_path)
    print(f"✅ 合成完成：{output_path}")
