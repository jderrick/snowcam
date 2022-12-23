# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import lib
import cv2
import json
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from PIL import Image

try:
    cfg = json.load(open('config.json'))
except Exception as e:
    print(e)
    exit(1)


class Weather:
    def __init__(self, city):
        self.latitude = cfg['resorts'][city]['latitude']
        self.longitude = cfg['resorts'][city]['longitude']
        self.weather = None
        self.hourly = None
        self.time_offset = int((time.timezone if (time.localtime().tm_isdst == 0) else time.altzone) / 60 / 60)
        self.im_scale = 5
        self.im_height = 480 * self.im_scale
        self.im_width = 160 * self.im_scale
        self.im = np.zeros([self.im_height, self.im_width, 3], np.uint8)
        self.im.fill(255)
        cv2.rectangle(self.im, (0, 0), (self.im_width, self.im_height), (0, 0, 0), self.im_scale)

    def load_weather(self):
        r = requests.get(f'https://api.open-meteo.com/v1/forecast?current_weather=true&'
                         f'hourly=snowfall,snow_depth,temperature_2m,windspeed_10m,winddirection_10m&'
                         f'latitude={self.latitude}&longitude={self.longitude}')
        if r.status_code == 200:
            data = r.json()
            self.weather = data['current_weather']
            # Default units of:
            # time: iso8601
            # snowfall: cm
            # snow_depth: m
            # temperature_2m: °C
            # windspeed_10m: km / h
            # winddirection_10m: °
            columns = ['time', 'snowfall', 'snow_depth',
                       'temperature_2m', 'windspeed_10m', 'winddirection_10m']
            self.hourly = pd.DataFrame(np.zeros((4, 6)), columns=columns)

            # Starting at 6AM + timezone; 6, 9, 12, 3pm
            for col in columns:
                for i in range(0, 4):
                    self.hourly.at[i, col] = data['hourly'][col][self.time_offset + 6 + (3 * i)]

    def create_image(self):
        for i in range(1, 4):
            y_pos = i * 600 - 20
            cv2.rectangle(self.im, (0, y_pos), (self.im_width, y_pos + 10), (0, 0, 0), -1)

        for i in range(0, 4):
            pos_x = 60
            pos_y = 70
            clock = datetime.fromisoformat(self.hourly['time'][i]) - timedelta(hours=self.time_offset)
            clock = clock.strftime('%H:%M')
            self.im = lib.add_text(self.im, xy=(pos_x, pos_y + i * 600), text=clock, align='left', anchor='lt',
                                   fill='black', stroke_fill='black', stroke_width=6,
                                   font='LiberationSerif-Regular.ttf', font_size=240)

            temperature_c = int(self.hourly['temperature_2m'][i])
            temperature_f = int((temperature_c * 9 / 5) + 32)
            self.im = lib.add_text(self.im, xy=(pos_x, pos_y + 220 + i * 600),
                                   text=f'{temperature_f}F/{temperature_c}C',
                                   align='left', anchor='lt',
                                   fill='black', stroke_fill='black', stroke_width=2,
                                   font='LiberationSerif-Regular.ttf', font_size=144)

            windspeed_k = int(self.hourly['windspeed_10m'][i])
            windspeed_m = int(windspeed_k * 0.6214)
            # winddirection = self.winddirection(self.hourly['winddirection_10m'][i])
            self.im = lib.add_text(self.im, xy=(pos_x, pos_y + 370 + i * 600),
                                   text=f'{windspeed_m}MPH/{windspeed_k}KPH', align='left', anchor='lt',
                                   fill='black', stroke_fill='black', stroke_width=2,
                                   font='LiberationSerif-Regular.ttf', font_size=110)

    def show(self):
        Image.fromarray(self.im).show()


def get_weather_image(city):
    w = Weather(city)
    w.load_weather()
    w.create_image()
    return w.im


