# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
