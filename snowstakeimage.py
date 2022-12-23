# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import lib
import cv2
import json
import math
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from PIL import Image


class SnowStakeImage:
    def __init__(self, im, resort):
        try:
            self.cfg = json.load(open('config.json'))['resorts'][resort]
            self.resort = resort
        except Exception as e:
            print(e)
            exit(1)

        self.im = im
        if self.im is None:
            self.im = self.retrieve_image()
        self.im = np.array(self.im)
        self.im = cv2.resize(self.im, None, fx=5, fy=5)
        self.rotation = 0
        self.inches = 0
        self.box_height = self.cfg['box']['height']
        self.box_width = self.cfg['box']['width']
        self.box_ypos = 0
        self.box_xpos = 0
        self.boxes = {}

    def retrieve_image(self):
        if self.resort == 'breckenridge' or self.resort == 'vail':
            url_type = 'timecam'

        if url_type == 'timecam':
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            # Round down to nearest 5 minutes and subtract another 5 minutes for upload cushion
            prev15 = now - timedelta(minutes=now.minute % 5) - timedelta(minutes=5)

            url = self.cfg['url']
            hour = prev15.strftime('%Y_%m_%d_%H')
            minute = hour + prev15.strftime('_%M')
            url = url.replace('##HOUR##', hour)
            url = url.replace('##MINUTE##', minute)
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                r.raw.decode_content = True
                return Image.open(r.raw)
            raise requests.RequestException
        else:
            raise ValueError(f'unsupported resort {self.resort}')

    def create_box(self, n):
        return ((self.box_xpos, self.box_ypos + self.box_height * n),
                (self.box_xpos + self.box_width, self.box_ypos + self.box_height * (n + 1)))

    def create_boxes(self):
        for i in range(0, self.cfg['inches'] + 1):
            self.boxes[str(self.cfg['inches'] - i)] = self.create_box(i)

    def rotate(self, rotation=0):
        if rotation != 0:
            rows, cols, _ = self.im.shape
            matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
            self.im = cv2.warpAffine(self.im, matrix, (cols, rows))

    def draw_boxes(self):
        for i in self.boxes:
            cv2.rectangle(self.im, self.boxes[i][0], self.boxes[i][1], (0, 255, 0), 5)

    def draw_snowline(self):
        box = self.boxes[str(self.inches)]
        spos_x = box[0][0]
        epos_x = spos_x - 300
        spos_y = box[0][1] + self.box_height
        epos_y = spos_y - 5
        cv2.rectangle(self.im, (spos_x, spos_y), (epos_x, epos_y), (0, 0, 0), -1)
        self.im = lib.add_text(self.im, xy=(epos_x, epos_y - 10), text=f'{self.inches}"', align='left', anchor='ls',
                               fill='white', stroke_fill='black', stroke_width=8,
                               font='LiberationSerif-Regular.ttf', font_size=200)

    def attach_header_text(self, ypos, xpos, text, align, anchor, font_size):
        self.im = lib.add_text(self.im, xy=(xpos, ypos), text=text, align=align, anchor=anchor,
                               fill='white', stroke_fill='black', stroke_width=8,
                               font='LiberationSerif-Regular.ttf', font_size=font_size)

    # Iteratively resize until the corners are non-black
    # Then use the iteration count to resize the original
    # Clunky but still pretty fast
    def resize_and_crop(self):
        im = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)
        height, width = im.shape
        iterations = 0
        while im[0][0] == 0 or im[height - 1][0] == 0 or \
                im[0][width - 1] == 0 or im[height - 1][width - 1] == 0:
            im = cv2.resize(im, None, fx=1.01, fy=1.01)
            new_height, new_width = im.shape
            y_offset = int((new_height - height) / 2)
            x_offset = int((new_width - width) / 2)
            im = im[y_offset:y_offset + height, x_offset:x_offset + width]
            iterations += 1
        scale = math.pow(1.01, iterations)
        self.im = cv2.resize(self.im, None, fx=scale, fy=scale)
        new_height, new_width, _ = self.im.shape
        y_offset = int((new_height - height) / 2)
        x_offset = int((new_width - width) / 2)
        self.im = self.im[y_offset:y_offset + height, x_offset:x_offset + width]

    def show(self):
        Image.fromarray(self.im).show()

    def auto_adjust(self):
        raise NotImplemented('auto_adjust must be implemented by subclass')

    def detect_snow(self):
        raise NotImplemented('detect_snow must be implemented by subclass')
