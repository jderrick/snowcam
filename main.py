# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import time
import weather
import vail
from PIL import Image
import numpy as np


def get_snow_image(city):
    if city == 'vail':
        return vail.get_vail_image()
    return None


def combine_images(left, right):
    left_height, left_width, _ = left.shape
    right_height, right_width, _ = right.shape
    combined_im = np.zeros([left_height, left_width + right_width, 3], np.uint8)
    combined_im.fill(255)
    combined_im[0:left_height, 0:left_width, :] = left[:, :, :]
    combined_im[0:left_height, left_width:left_width + right_width, :] = right[:, :, :]
    return combined_im


if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        snow_im = get_snow_image('vail')
        weather_im = weather.get_weather_image('vail')
        im = combine_images(left=weather_im, right=snow_im)
        Image.fromarray(im).show()
        exit(0)
        time.sleep(300 - ((time.time() - start_time) % 300))
