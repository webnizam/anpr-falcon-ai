from datetime import datetime
import glob
import json
import os
import string
import sys
import time
import cv2
import easyocr
from fai_anpr_model_loader import FaiAnprModelLoader
from fai_threaded_camera import WebcamVideoStream


class FaiNumberPlateFetcher:

    def __init__(self, capture_device, model_path=r"best.pt"):
        self.allowlist = string.digits + string.ascii_letters
        self.capture_device = capture_device
        self.text_font = cv2.FONT_HERSHEY_PLAIN
        self.color = (0, 0, 255)
        self.text_font_scale = 1.25
        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.last_auth_time = None
        self.last_auth_read_count = 0
        self.authorized = False
        self.auth_timeout_seconds = 5
        self.auth_read_threshold = 5

        self.authorized_plates = self.get_authorized_plates()
        print(f'{self.authorized_plates=}')

        cpu_or_cuda = "mps" if sys.platform == 'darwin' else 'cuda'
        self.anpr = FaiAnprModelLoader(model_path, cpu_or_cuda)
        self.ocr = easyocr.Reader(['en'], gpu=True)
        try:
            self.streamer = WebcamVideoStream(self.capture_device)
        except Exception as e:
            print('Its the error', e, sep='\n\n')
            exit(0)

    def get_Text(self, result):
        try:
            txts = [line[1] for line in result if len(line[1]) < 6]
            if len(txts) > 4:
                txts = txts[:3]

            return ' '.join(txts)
        except Exception as e:
            print(e)
            return ""

    def resize_image(self, image, height=400):
        aspect_ratio = float(image.shape[1])/float(image.shape[0])
        window_width = height/aspect_ratio
        image = cv2.resize(image, (int(height), int(window_width)),
                           interpolation=cv2.INTER_LANCZOS4)
        return image

    def remove_img(self, img_name):
        os.remove(img_name)

    def start(self):

        self.streamer.start()
        last_time = datetime.now()

        ctr = 0
        start = time.perf_counter()

        while True:
            ctr = ctr + 1
            image = self.streamer.read()
            do_process = (datetime.now() - last_time).microseconds/1000 > 100
            if image is not None:
                if do_process:
                    last_time = datetime.now()

                    result = self.anpr.get_number_plates(image)
                    start = time.perf_counter()
                    if len(result) > 0:
                        for i in result:
                            p1 = (int(i[0]), int(i[1]))
                            p2 = (int(i[2]), int(i[3]))

                            text_origin = (int(i[0]), int(i[1])-3)
                            cv2.rectangle(
                                image, p1, p2, color=self.color, thickness=2)
                            img = cv2.rectangle(
                                image, p1, p2, color=self.color, thickness=2)
                            plate_id = self.get_bbox_content(img)
                            self.save_to_json(plate_id)
                            font_scale = self.get_optimal_font_scale(
                                plate_id, self.get_distance(
                                    p1, (int(i[2]), int(i[1])))
                            )
                            cv2.putText(
                                image,
                                text=plate_id,
                                org=text_origin,
                                fontFace=self.text_font,
                                fontScale=font_scale,
                                color=self.color,
                                thickness=2
                            )
                            for plate in self.authorized_plates:
                                if plate in plate_id:
                                    self.authorized = True
                                    self.last_auth_time = datetime.now()
                                    self.last_auth_read_count += 1
                                    # print(f'{last_auth_read_count=}')
                                    break

                if self.authorized and self.last_auth_time \
                    and (datetime.now() - self.last_auth_time).total_seconds() < self.auth_timeout_seconds \
                        and self.last_auth_read_count > self.auth_read_threshold:

                    print('\n\n')

                    print(f'{self.authorized=}')
                    print(
                        f'{(datetime.now() - self.last_auth_time).total_seconds()=}')
                    print(f'{self.auth_timeout_seconds=}')
                    print(f'{self.last_auth_read_count=}')
                    print(f'{self.auth_read_threshold=}')

                    self.put_auth_stat(image, True)
                else:
                    self.put_auth_stat(image, False)

                if self.last_auth_time and (datetime.now() - self.last_auth_time).total_seconds() > self.auth_timeout_seconds:
                    self.last_auth_read_count = 0
                    self.authorized = False

                # Add our Company
                cv2.putText(
                    image,
                    text='FALCONS.AI',
                    org=(10, 70),
                    fontFace=self.text_font,
                    fontScale=3,
                    color=(0, 0, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                self.new_frame_time = time.time()
                fps = 1/(self.new_frame_time-self.prev_frame_time)
                self.prev_frame_time = self.new_frame_time
                fps = int(fps)
                fps = str(fps)
                text_size = cv2.getTextSize(fps, font, 3, 3)[0]

                cv2.putText(image, fps, (image.shape[1]-(text_size[0]+7), 115), font, 3,
                            (100, 255, 0), 3, cv2.LINE_AA)

                elapsed = time.perf_counter() - start
                cv2.putText(
                    image,
                    text=f'{round(elapsed, 2)} seconds',
                    org=(10, 105),
                    fontFace=self.text_font,
                    fontScale=2,
                    color=(90, 85, 68),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow("ANPR - FALCONS.AI", image)
                print(f'Took {elapsed} seconds to process the frame.')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.streamer.stop()
        self.streamer.destroy()
        cv2.destroyAllWindows()

    def get_bbox_content(self, img):
        result = self.ocr.readtext(
            self.resize_image(img),
            allowlist=self.allowlist
        )
        plate_num = self.get_Text(result)
        return plate_num

    def get_optimal_font_scale(self, text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(
                text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/5
        return 1

    def get_distance(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def remove_previous_files(self):
        files = glob.glob('./plate_capture/*')
        for f in files:
            print(f)
            os.remove(f)

    def get_authorized_plates(self):
        auth_plates = []
        with open('authorized_plates.json') as f:
            try:
                auth_plates = json.load(f)
                if not auth_plates:
                    auth_plates = []
            except Exception as e:
                print(e)
                auth_plates = []
        return auth_plates

    def save_to_json(self, plate):
        if plate:
            with open('result.json') as f:
                try:
                    data = json.load(f)
                    if not data:
                        data = []
                except:
                    data = []
            data.append(plate)
            data = list(set(data))
            with open('result.json', 'w') as f:
                json.dump(data, f, indent=4,
                          separators=(',', ': '))

    def put_auth_stat(self, image, authorized):
        height_, width_, _ = image.shape

        if authorized:
            text_ = 'Authorized'
            text_size_ = cv2.getTextSize(
                text_, self.text_font, 5, 2)[0]
            origin_ = (
                int((width_-text_size_[0])/2), height_ - text_size_[1])
            color_ = (0, 255, 0)
        else:
            text_ = 'Not Authorized'
            text_size_ = cv2.getTextSize(
                text_, self.text_font, 5, 2)[0]
            origin_ = (
                int((width_-text_size_[0])/2), height_ - text_size_[1])
            color_ = (0, 0, 255)

        cv2.putText(
            image,
            text=text_,
            org=origin_,
            fontFace=self.text_font,
            fontScale=5,
            color=color_,
            thickness=3
        )
