# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import weather
import vail
import numpy as np
from PIL import Image


cfg = None
try:
    cfg = json.load(open('config.json'))
except Exception as e:
    print(e)
    exit(1)


resorts = list(cfg['resorts'])
resort_index = 0


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


def get_next_resort_image():
    global resort_index
    snow_im = get_snow_image(resorts[resort_index])
    weather_im = weather.get_weather_image(resorts[resort_index])
    combined_im = combine_images(left=weather_im, right=snow_im)
    resort_index += 1
    if resort_index == len(resorts):
        resort_index = 0
    return combined_im


if __name__ == '__main__':
    im = get_next_resort_image()
    Image.fromarray(im).show()
