# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def reject_outliers(data, m=6.):
    if type(data) is list:
        data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()


def add_text(im, xy, text, font=None, font_size=15, fill='black',
             stroke_fill=None, stroke_width=0, embedded_color=False,
             anchor='ms', align='left', spacing=4, direction='ltr'):
    pil_im = Image.fromarray(im)
    if font:
        font = ImageFont.truetype(font, font_size)
    else:
        font = ImageFont.truetype('NotoSans-Regular.ttf', font_size)
    draw = ImageDraw.Draw(pil_im)
    draw.text(xy=xy, text=text, fill=fill, font=font, stroke_fill=stroke_fill, stroke_width=stroke_width,
              embedded_color=embedded_color, anchor=anchor, align=align, spacing=spacing, direction=direction)
    return np.array(pil_im)


def combine_images(left, right):
    left_height, left_width, _ = left.shape
    right_height, right_width, _ = right.shape
    combined_im = np.zeros([left_height, left_width + right_width, 3], np.uint8)
    combined_im.fill(255)
    combined_im[0:left_height, 0:left_width, :] = left[:, :, :]
    combined_im[0:left_height, left_width:left_width + right_width, :] = right[:, :, :]
    return combined_im
