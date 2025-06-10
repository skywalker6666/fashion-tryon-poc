import cv2
import numpy as np


def warp_clothing_onto_person(clothes_path, person_path, output_path, src_points, dst_points):
    # 讀取衣服圖和人物圖（RGBA）
    clothes_img = cv2.imread(clothes_path, cv2.IMREAD_UNCHANGED)
    person_img = cv2.imread(person_path, cv2.IMREAD_UNCHANGED)

    # 檢查衣服圖有無 alpha 通道
    if clothes_img.shape[2] != 4:
        raise ValueError("Clothing image must have an alpha channel (RGBA).")
    if person_img.shape[2] != 4:
        # 自動補上 alpha 通道
        person_img = cv2.cvtColor(person_img, cv2.COLOR_RGB2RGBA)
    # 將 src/dst 轉為 np.float32
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    # 計算透視變換矩陣並進行 warping
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(clothes_img, matrix, (person_img.shape[1], person_img.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)

    # 分離 alpha 與 RGB
    warped_rgb = warped[:, :, :3].astype(float)
    warped_alpha = warped[:, :, 3].astype(float) / 255.0
    person_rgb = person_img[:, :, :3].astype(float)

    # alpha blending 疊圖
    blended_rgb = warped_rgb * warped_alpha[..., None] + person_rgb * (1 - warped_alpha[..., None])
    blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)

    # 回存為 PNG（加回 alpha，這裡直接用 person 的 alpha）
    blended_rgba = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2RGBA)
    blended_rgba[:, :, 3] = person_img[:, :, 3]  # 如果你有原 alpha，可替代或保留

    cv2.imwrite(output_path, blended_rgba)
