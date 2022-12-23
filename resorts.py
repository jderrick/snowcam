# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import lib
import weather
import json
from vail import vail
from breckenridge import breckenridge
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
    try:
        return eval(city).get_image()
    except Exception as e:
        raise e


def get_next_resort_image():
    global resort_index
    # resort = resorts[resort_index]
    resort = 'vail'
    snow_im = get_snow_image(resort)
    weather_im = weather.get_weather_image(resort)
    combined_im = lib.combine_images(left=weather_im, right=snow_im)
    resort_index += 1
    if resort_index == len(resorts):
        resort_index = 0
    return combined_im


if __name__ == '__main__':
    im = get_next_resort_image()
    Image.fromarray(im).show()
