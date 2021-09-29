import glob
import PIL
from PIL import ImageEnhance, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch import nn


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    width, height = img.size
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0) 
    pad_len = int(abs(width - height) / 2)  
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


def jp_unicode_decimals():

    kanji = list(range(int(0x4e00), int(0x9faf)+1))
    punc = list(range(int(0x3000), int(0x303f)+1))
    hiragana = list(range(int(0x3040), int(0x309f)+1))
    katakana = list(range(int(0x30a0), int(0x30ff)+1))

    unicode_dec = sum([kanji, punc, hiragana, katakana], [])

    return unicode_dec


if __name__ == '__main__':

    font_path = '/mnt/data01/otf/NotoSerifCJKjp-Regular.otf'
    save_path = '/mnt/data01/rendered_chars/all_chars_NotoSerifCJKjp-Regular'

    digital_font = ImageFont.truetype(font_path, size=256)
    os.makedirs(save_path, exist_ok=True)

    uni_dec = jp_unicode_decimals()

    for idx, i in tqdm(enumerate(uni_dec), total=len(uni_dec)):
        render_char = draw_single_char(chr(i), digital_font, 256)
        if render_char is not None:
            render_char.resize((64,64)).save(os.path.join(save_path, f'{chr(i)}_{idx}.png'))