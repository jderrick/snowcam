import time
import requests
import cv2
import numpy as np
import json
from datetime import datetime, timedelta, timezone
from PIL import Image

try:
    cfg = json.load(open('config.json'))['vail']
except Exception as e:
    print(e)
    exit(1)


class SnowStakeImage:
    class Box:
        def __init__(self,
                     y_adjust=cfg['box']['y_adjust'],
                     x_adjust=cfg['box']['x_adjust']):
            self.height = cfg['box']['height']
            self.width = cfg['box']['width']
            self.y_adjust = y_adjust
            self.x_adjust = x_adjust
            self.pos_y = cfg['box']['pos_y'] + self.y_adjust
            self.spos_x = cfg['box']['spos_x'] + self.x_adjust
            self.epos_x = cfg['box']['epos_x'] + self.x_adjust

        def create_box(self, n):
            return {'left': ((self.spos_x - self.width, self.pos_y + self.height * n),
                             (self.spos_x, self.pos_y + self.height * (n + 1))),
                    'right': ((self.epos_x, self.pos_y + self.height * n),
                              (self.epos_x + self.width, self.pos_y + self.height * (n + 1)))}

    def __init__(self, im, rotation=cfg['rotation'],
                 y_adjust=cfg['box']['y_adjust'],
                 x_adjust=cfg['box']['x_adjust']):
        self.im = np.array(im)
        self.im = cv2.resize(self.im, None, fx=5, fy=5)
        self.im = self.rotate(rotation)
        self.inches = 0
        self.box = self.Box(y_adjust=y_adjust, x_adjust=x_adjust)
        self.boxes = {}
        for i in range(0, 19):  # 0-18 inclusive
            self.boxes[str(18 - i)] = self.box.create_box(i)

    def rotate(self, rotation=0):
        if rotation == 0:
            return self.im
        rows, cols, _ = self.im.shape
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        return cv2.warpAffine(self.im, matrix, (cols, rows))

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
            for side in ['left', 'right']:
                (spos, epos) = (self.boxes[str(i)][side][0], self.boxes[str(i)][side][1])
                box = self.im[spos[1]:epos[1], spos[0]:epos[0]]
                hsv = cv2.cvtColor(box, cv2.COLOR_RGB2HSV)

                # Below 6", check count of hsv pink pixels
                if int(i) < 6:
                    mask = cv2.inRange(hsv, lower_pink, upper_pink)
                    if np.count_nonzero(mask) > 1000:
                        self.inches = i
                        return i

                # Check count of im blue pixels
                elif int(i) >= 6:
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    if np.count_nonzero(mask) >= 10:
                        self.inches = i
                        return i

    def draw_boxes(self):
        for i in self.boxes:
            for side in ['left', 'right']:
                cv2.rectangle(self.im, self.boxes[i][side][0], self.boxes[i][side][1], (0, 255, 0), 5)

    def draw_snowline(self):
        box = self.boxes[str(self.inches)]
        spos_x = box['left'][0][0]
        epos_x = spos_x - self.box.width * 2
        spos_y = box['left'][0][1] + self.box.height
        epos_y = spos_y - 5
        cv2.rectangle(self.im, (spos_x, spos_y), (epos_x, epos_y), (0, 0, 0), -1)
        self.im = cv2.putText(self.im, f'{self.inches}"', org=(epos_x, epos_y - 10),
                              fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=4,
                              color=(0, 0, 0), thickness=8, lineType=cv2.LINE_AA)

    def show(self):
        Image.fromarray(self.im).show()



def retrieve_vail_image():
    #Vail url requires the year, month, day, hour, and rounded minute
    #https://terra.timecam.tv/express/mediablock/timestreams/vailresort/vail-official-snow-stake~640/hour/2022_12_14_13/vail-official-snow-stake~640_2022_12_14_13_15_00.jpg
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


def create_snow_stake_image(im):
    s = SnowStakeImage(im)
    return s


def run_vail():
    # Raw Vail image
    # Parameters change when camera is adjusted
    s = SnowStakeImage(retrieve_vail_image())

    # No-snow reference image
    # s = SnowStakeImage(Image.open('vail-official-snow-stake~640_2022_12_14_13_10_00_00.jpg'))

    # 10" snow reference image
    # s = SnowStakeImage(Image.open('vail-official-snow-stake~640_2022_12_14_13_05_00_00.jpg'),
    #                    x_adjust = -25, y_adjust = 5, rotation = -0.5)

    inches = s.detect_snow()
    # print(f'{datetime.now()} Inches: {inches}')
    # s.draw_boxes()
    s.draw_snowline()
    s.show()


if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        run_vail()
        time.sleep(300 - ((time.time() - start_time) % 300))
