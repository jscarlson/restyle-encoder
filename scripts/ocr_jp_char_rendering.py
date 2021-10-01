import glob
import PIL
from PIL import ImageEnhance, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch import nn

from fontTools.ttLib import TTFont
from itertools import chain
from fontTools.unicode import Unicode
from collections import defaultdict


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


def get_unicode_coverage_from_ttf(ttf_path):
    with TTFont(ttf_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1) as ttf:
        chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
        chars_dec = [x[0] for x in chars]
        return chars_dec, [chr(x) for x in chars_dec]


def filter_recurring_hash(charset, font, canvas_size):
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size)
        if img is not None:
            hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


if __name__ == '__main__':

    font_paths = (
        '/mnt/data01/otf/NotoSerifCJKjp-Regular.otf',
        '/mnt/data01/ttf/HinaMincho-Regular.ttf',
        '/mnt/data01/ttf/NewTegomin-Regular.ttf',
    )
    save_path = '/mnt/data01/rendered_chars/joyo_chars_many_renders'
    os.makedirs(save_path, exist_ok=True)

    uni_dec = jp_unicode_decimals()
    # with open("/mnt/data01/charsets/joyo_kanji.txt") as f:
    #     uni_dec = [ord(c) for c in f.read().split()]

    idx = 0
    for font_path in font_paths:

        digital_font = ImageFont.truetype(font_path, size=256)
        _, covered_chars = get_unicode_coverage_from_ttf(font_path)
        covered_chars_kanji_plus = list(set([c for c in covered_chars if ord(c) in uni_dec]))

        filter_hashes = set(filter_recurring_hash(covered_chars_kanji_plus, digital_font, 256))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

        for c in tqdm(covered_chars_kanji_plus, total=len(covered_chars_kanji_plus)):
            render_char = draw_single_char(c, digital_font, 256)
            if render_char is None:
                continue
            render_hash = hash(render_char.tobytes())
            if render_hash in filter_hashes:
                continue
            render_char.resize((64,64)).save(os.path.join(save_path, f'{c}_{idx}.png'))
            idx += 1