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


class Vail(SnowStakeImage):
    def __init__(self, im):
        super().__init__(im, 'vail')

    # Vail has a white image at the top of the snow stake
    # And most of the rest of the stake is blue
    def auto_adjust(self):
        ypos = 200
        xpos = 1100
        height = 1000
        width = 500

        crop_im = self.im[ypos:ypos + height, xpos:xpos + width]
        crop_im = cv2.cvtColor(crop_im, cv2.COLOR_RGB2GRAY)
        th, im_crop = cv2.threshold(crop_im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        stride = int(height / 8) - 1
        dist = []
        for y in range(0, height, stride):
            for x in range(0, width):
                # This is very clearly Binary thresholded to black-and-white
                # Why does it still require a grayscale value?
                if crop_im[y][x] > 100:
                    dist.append((y, x))
                    break
        xval = int(mean(lib.reject_outliers([x[1] for x in dist])))
        self.box_xpos = xval + xpos + 5

        # Trig: tan N = opposite / adjacent
        rotations = []
        for i in range(0, len(dist) - 1):
            oa = (dist[i + 1][1] - dist[i][1]) / (dist[i + 1][0] - dist[i][0])
            rotations.append(math.atan(oa) * 180 / math.pi)
        self.rotation = -mean(lib.reject_outliers(rotations))
        self.rotate(self.rotation)

        # Now determine x/y adjust
        ypos = 200
        xpos = self.box_xpos + 10
        height = 800
        width = 500 - xval

        im = cv2.cvtColor(self.im, cv2.COLOR_RGB2HSV)
        crop_im = im[ypos:ypos + height, xpos:xpos + width]
        lower_pink = np.array([100, 0, 0])
        upper_pink = np.array([180, 255, 255])
        mask = cv2.inRange(crop_im, lower_pink, upper_pink)
        for y in range(0, height):
            if mask[y][0] == 255:
                self.box_ypos = y + ypos
                break

    def detect_snow(self):
        # Below 6", need to check snow stake consistency
        # Snow adds color consistency creating a solid red hsv layer
        # Non-snow images have inconsistencies creating red and pink sections
        # Pinkish should be lower-bounded to about 140, but for some reason it needs 100
        lower_pink = np.array([100, 0, 0])
        upper_pink = np.array([180, 255, 255])

        # 6" and above, stake is blue and image can be compared against blue
        lower_blue = np.array([90, 127, 127])
        upper_blue = np.array([130, 255, 255])
        for i in range(len(self.boxes)):
            (spos, epos) = (self.boxes[str(i)][0], self.boxes[str(i)][1])
            box = self.im[spos[1]:epos[1], spos[0]:epos[0]]
            hsv = cv2.cvtColor(box, cv2.COLOR_RGB2HSV)

            # Below 6", check count of hsv pink pixels
            if int(i) < 6:
                mask = cv2.inRange(hsv, lower_pink, upper_pink)
                if np.count_nonzero(mask) > 1000:
                    self.inches = i
                    return i

            # 6" and above, check count of im blue pixels
            elif int(i) >= 6:
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                if np.count_nonzero(mask) >= 10:
                    self.inches = i
                    return i

    def draw_boxes(self):
        for i in self.boxes:
            cv2.rectangle(self.im, self.boxes[i][0], self.boxes[i][1], (0, 255, 0), 5)


def run_vail():
    im = None

    # No-snow reference image
    # im = Image.open(join('vail', 'vail-official-snow-stake~640_2022_12_14_13_10_00_00.jpg'))

    # 10" snow reference image
    # im = Image.open(join('vail', 'vail-official-snow-stake~640_2022_12_14_13_05_00_00.jpg'),
    #                 x_adjust=-25, y_adjust=5, rotation=-0.5)

    s = Vail(im)
    s.auto_adjust()
    s.create_boxes()
    s.detect_snow()
    # s.draw_boxes()
    s.draw_snowline()
    s.resize_and_crop()
    s.attach_header_text(300, 3100, f'VAIL\n{s.inches}"', 'right', 'rs', 400)
    return s


def get_image():
    s = run_vail()
    return s.im


if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        s = run_vail()
        s.show()
        time.sleep(300 - ((time.time() - start_time) % 300))
