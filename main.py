# Copyright (c) 2022, Jonathan Derrick
# SPDX-License-Identifier: GPL-3.0-or-later
import time
import resorts
from PIL import Image


if __name__ == '__main__':
    # Runs every minute, avoiding drift
    start_time = time.time()
    while True:
        im = resorts.get_next_resort_image()
        Image.fromarray(im).show()
        time.sleep(60 - ((time.time() - start_time) % 60))
