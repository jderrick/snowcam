# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import lib
import text
import math
import time
import json
import requests
import cv2
import numpy as np
from os.path import join
from statistics import mean
from datetime import datetime, timedelta, timezone
from PIL import Image

try:
    cfg = json.load(open('config.json'))['resorts']['breckenridge']
except Exception as e:
    print(e)
    exit(1)


class SnowStakeImage:
    def __init__(self, im):
        self.im = np.array(im)
        self.im = cv2.resize(self.im, None, fx=5, fy=5)
        self.rotation = 0
        self.inches = 0
        self.box_height = cfg['box']['height']
        self.box_width = cfg['box']['width']
        self.box_ypos = cfg['box']['ypos']
        self.box_xpos = cfg['box']['xpos']
        self.boxes = {}

    def create_box(self, n):
        return ((self.box_xpos, self.box_ypos + self.box_height * n),
                (self.box_xpos + self.box_width, self.box_ypos + self.box_height * (n + 1)))

    def create_boxes(self):
        for i in range(0, cfg['inches'] + 1):  # 0-24 inclusive
            self.boxes[str(cfg['inches'] - i)] = self.create_box(i)

    @staticmethod
    def rotate(im, rotation=0):
        if rotation == 0:
            return im
        rows, cols, _ = im.shape
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        return cv2.warpAffine(im, matrix, (cols, rows))

    def detect_snow(self):
        # Snow adds color consistency creating a solid blue hsv layer
        # Non-snow images have inch markers in green in the hsv layer
        lower_green = np.array([45, 0, 0])
        upper_green = np.array([180, 255, 255])
        for i in range(len(self.boxes)):
            (spos, epos) = (self.boxes[str(i)][0], self.boxes[str(i)][1])
            box = self.im[spos[1]:epos[1], spos[0]:epos[0]]
            mask = cv2.inRange(box, lower_green, upper_green)
            # Image.fromarray(mask).show()
            # time.sleep(1)
            # Image.fromarray(box).show()
            # time.sleep(1)
            # print(i, np.count_nonzero(mask))
            if np.count_nonzero(mask) > 100:
                self.inches = i
                return i

    def auto_adjust(self):
        im = cv2.cvtColor(self.im, cv2.COLOR_RGB2HLS)
        lower = np.uint8([0, 200, 0])
        upper = np.uint8([255, 255, 255])
        mask = cv2.inRange(im, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        im = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Crop can't be too tall as snow might interfere
        ypos = 350
        xpos = 1300
        height = 300
        width = 500
        crop_im = im[ypos:ypos + height, xpos:xpos + width]
        crop_im = cv2.dilate(crop_im, kernel, iterations=4)

        stride = int(height / 8) - 1
        dist = []
        for y in range(0, height, stride):
            for x in range(0, width):
                if crop_im[y][x] == 255:
                    dist.append((y, x))
                    break

        # Trig: tan N = opposite / adjacent
        rotations = []
        for i in range(0, len(dist) - 1):
            oa = (dist[i + 1][1] - dist[i][1]) / (dist[i + 1][0] - dist[i][0])
            rotations.append(math.atan(oa) * 180 / math.pi)
        self.rotation = -mean(lib.reject_outliers(rotations))
        self.im = self.rotate(self.im, self.rotation)

        # Now determine x/y adjust
        ypos = 200
        xpos = 1300
        height = 300
        width = 500
        im = cv2.cvtColor(self.im, cv2.COLOR_RGB2HLS)
        mask = cv2.inRange(im, lower, upper)
        im = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        crop_im = im[ypos:ypos + height, xpos:xpos + width]
        crop_im = cv2.dilate(crop_im, kernel, iterations=4)

        # 2-pass search
        # 1st pass, find top of stake (or snow)
        x = int(width / 2)
        for y in range(0, height):
            if crop_im[y][x] == 255:
                self.box_ypos = ypos + y
                break

        # 2nd pass, find left side of stake
        height, width = crop_im.shape
        stride = int((height - y) / 8) - 1
        xvals = []
        for y in range(y, height, stride):
            for x in range(0, width):
                if crop_im[y][x] == 255:
                    xvals.append(x)
                    break
        x = int(mean(lib.reject_outliers(xvals)))
        self.box_xpos = x + xpos

        # Find top of stake from left
        for y in range(0, height):
            if crop_im[y][x + 5] == 255:
                self.box_ypos = ypos + y
                break

    def draw_boxes(self):
        for i in self.boxes:
            cv2.rectangle(self.im, self.boxes[i][0], self.boxes[i][1], (0, 255, 0), 5)

    def draw_snowline(self):
        box = self.boxes[str(self.inches)]
        spos_x = box[0][0]
        epos_x = spos_x - self.box_width * 2
        spos_y = box[0][1] + self.box_height
        epos_y = spos_y - 5
        cv2.rectangle(self.im, (spos_x, spos_y), (epos_x, epos_y), (0, 0, 0), -1)
        self.im = text.add_text(self.im, xy=(epos_x, epos_y - 10), text=f'{self.inches}"', align='left', anchor='ls',
                                fill='white', stroke_fill='black', stroke_width=8,
                                font='LiberationSerif-Regular.ttf', font_size=200)

    def attach_header_text(self):
        ypos = 20
        xpos = self.im.shape[1] - 100
        self.im = text.add_text(self.im, xy=(xpos, ypos), text=f'{self.inches}"', align='right', anchor='rt',
                                fill='white', stroke_fill='black', stroke_width=8,
                                font='LiberationSerif-Regular.ttf', font_size=500)

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


def retrieve_breckenridge_image():
    # Breckenridge url is the same format as Vail
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # Round down to nearest 5 minutes and subtract another 5 minutes for upload cushion
    prev15 = now - timedelta(minutes=now.minute % 5) - timedelta(minutes=5)

    url = cfg['url']
    hour = prev15.strftime('%Y_%m_%d_%H')
    minute = hour + prev15.strftime('_%M')
    url = url.replace('##HOUR##', hour)
    url = url.replace('##MINUTE##', minute)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        return Image.open(r.raw)
    raise requests.RequestException


def run_breckenridge():
    s = SnowStakeImage(retrieve_breckenridge_image())

    # No-snow reference image
    # s = SnowStakeImage(Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_20_04_50_00_00.jpg')))

    # Highly tilted reference image
    # Ideal for tilt correction testing
    # s = SnowStakeImage(Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_19_04_50_00_00.jpg')))

    # 3-inches snow reference image
    # s = SnowStakeImage(Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_21_12_05_00_00.jpg')))

    s.auto_adjust()
    s.create_boxes()
    s.detect_snow()

    # For adjusting camera parameters
    # s.draw_boxes()

    s.draw_snowline()
    s.resize_and_crop()
    # s.show()
    return s


def get_image():
    s = run_breckenridge()
    s.attach_header_text()
    return s.im


if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        run_breckenridge()
        time.sleep(300 - ((time.time() - start_time) % 300))
