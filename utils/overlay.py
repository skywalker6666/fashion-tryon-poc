from PIL import Image


def overlay_clothes(user_path, clothes_path, output_path, position=(100, 150)):
    user_img = Image.open(user_path).convert("RGBA")
    clothes_img = Image.open(clothes_path).convert("RGBA")
    clothes_width = int(user_img.width * 0.5)
    clothes_height = int(clothes_img.height * 0.5)
    clothes_img = clothes_img.resize((clothes_width, clothes_height))

    user_img.paste(clothes_img, position, clothes_img)
    user_img.save(output_path)
