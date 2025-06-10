import sys
import torch
from PIL import Image
from collections import OrderedDict

# 加入 MODNet 模型路徑
sys.path.append('./modnet_src/src')
from models.modnet import MODNet
from torchvision import transforms

# ✅ 定義圖像預處理流程
img_preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 初始化模型（載入權重）
modnet = MODNet(backbone_pretrained=False)
# 載入原始權重
state_dict = torch.load('modnet_src/pretrained/modnet.ckpt', map_location='cpu')


# 處理 module. 前綴
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


cleaned_state_dict = remove_module_prefix(state_dict)

# 套用權重
modnet.load_state_dict(cleaned_state_dict, strict=False)
modnet.eval()


def segment_person(input_path, output_path):
    # 開啟圖片
    img = Image.open(input_path).convert('RGB')
    orig_size = img.size

    # 轉 tensor 並 resize
    img_tensor = img_preprocess(img).unsqueeze(0)

    # 推論 alpha matte
    with torch.no_grad():
        _, _, matte = modnet(img_tensor, True)

    matte = matte.squeeze().cpu().numpy()
    matte = Image.fromarray((matte * 255).astype('uint8')).resize(orig_size)

    # 合成 RGBA 圖
    rgb = img.convert('RGBA')
    alpha = matte
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    rgba.save(output_path)
    print(f"✅ 輸出透明人物圖：{output_path}")
