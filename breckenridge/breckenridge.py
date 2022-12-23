# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import lib
from snowstakeimage import SnowStakeImage
import cv2
import math
import time
import numpy as np
from os.path import join
from PIL import Image
from statistics import mean


class Breckenridge(SnowStakeImage):
    def __init__(self, im):
        super().__init__(im, 'breckenridge')

    def auto_adjust(self):
        im = cv2.cvtColor(self.im, cv2.COLOR_RGB2HLS)
        lower = np.uint8([0, 180, 0])
        upper = np.uint8([255, 255, 255])
        mask = cv2.inRange(im, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        im = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Crop can't be too tall as snow might interfere
        ypos = 200
        xpos = 1300
        height = 500
        width = 500
        crop_im = im[ypos:ypos + height, xpos:xpos + width]
        crop_im = cv2.dilate(crop_im, kernel, iterations=4)
        crop_im = cv2.erode(crop_im, kernel, iterations=8)

        stride = int(height / 12) - 1
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
        rotations = lib.reject_outliers(rotations)
        self.rotation = -mean(rotations)
        self.rotate(self.rotation)

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


def run_breckenridge():
    im = None

    # No-snow reference image
    # im = Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_20_04_50_00_00.jpg'))

    # Highly tilted reference image
    # Ideal for tilt correction testing
    # im = Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_19_04_50_00_00.jpg'))

    # 2-3 inch snow reference image
    # im = Image.open(join('breckenridge', 'breckenridge-snowstake-ca~640_2022_12_21_12_05_00_00.jpg'))

    s = Breckenridge(im)
    s.auto_adjust()
    s.create_boxes()
    s.detect_snow()
    # s.draw_boxes()
    s.draw_snowline()
    s.resize_and_crop()
    s.attach_header_text(20, s.im.shape[1] - 100, f'{s.inches}"','right', 'rt', 500)
    return s


def get_image():
    s = run_breckenridge()
    return s.im


if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        s = run_breckenridge()
        s.show()
        time.sleep(300 - ((time.time() - start_time) % 300))
