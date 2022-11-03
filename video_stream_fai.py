# Falcons.ai
# Michael Stattelman 2022
import glob
import os
import cv2
import time
import torch
import string
# import pytesseract
import numpy as np
from PIL import Image
from cv2 import waitKey
from datetime import datetime
import matplotlib.pyplot as plt
#from pytesseract import image_to_string
# from paddleocr import PaddleOCR,draw_ocr
import easyocr
import time
import json


from threaded_camera import WebcamVideoStream


def get_Text(result):
    try:
        # print(result)
        # result = result[0][1]
        # image = Image.open(img_path).convert('RGB')
        # boxes = [line[1] for line in result[0]]
        txts = [line[1] for line in result if len(line[1]) < 6]
        return ' '.join(txts)
    except Exception as e:
        print(e)
        return ""


def remove_img(img_name):
    os.remove(img_name)


# Extract the image from the bounding box
def get_bbox_content(img):
    # now = datetime.now()
    # t = now.strftime("%m-%d-%Y-%I-%M-%S")
    # filename = "plate_capture/"+str(t)+".jpg"
    # hsv_min = np.array([0, 250, 100], np.uint8)
    # hsv_max = np.array([10, 255, 255], np.uint8)
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # frame_threshed = cv2.inRange(hsv_img, hsv_min, hsv_max)

    # # Perform morphology
    # se = np.ones((1, 1), dtype='uint8')
    # image_close = cv2.morphologyEx(frame_threshed, cv2.MORPH_CLOSE, se)

    # # detect contours on the morphed image
    # ret, thresh = cv2.threshold(image_close, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(
    #     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # areaArray = []
    # for i, c in enumerate(contours):
    #     area = cv2.contourArea(c)
    #     areaArray.append(area)

    # # Sort countours based on area
    # sorteddata = sorted(zip(areaArray, contours),
    #                     key=lambda x: x[0], reverse=True)

    # # find the nth largest contour [n-1][1], in this case 2
    # largestcontour = sorteddata[0][1]

    # # get the bounding rectangle of the contour
    # x, y, w, h = cv2.boundingRect(largestcontour)

    # cropped_img = img[y+3:y+h-3, x+3:x+w-3]
    # up_width = 1024
    # up_height = 768
    # up_points = (up_width, up_height)
    # resized_up = cv2.resize(cropped_img, up_points,
    #                         interpolation=cv2.INTER_LINEAR)
    # # Convert to greyscale
    # #gray_image = cv2.cvtColor(resized_up, cv2.COLOR_BGR2GRAY)
    # # Reduce the noise
    # gray_image = cv2.bilateralFilter(resized_up, 11, 17, 17)

    # cv2.imwrite(filename, gray_image)
    # OCR with paddle more accurate than pytesseract

    dim = (600, 400)

    # img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    print(f'{img.shape=}')

    result = ocr.readtext(
        img,
        allowlist=allowlist
    )
    plate_num = get_Text(result)
    print(plate_num)
    # OCR with pytesseract
    #plate_num = pytesseract.image_to_string(gray_image, lang='eng')
    return plate_num


# Load Paddle Model
start = time.perf_counter()


# Model
model_path = r"best.pt"  # custom model path
video_path = r"videos/plates2.mp4"  # input video path
cpu_or_cuda = "mps"  # choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)
# Use local Yolo Model
model_name = 'best.pt'
model = torch.hub.load('ultralytics-yolov5-6371de8/', 'custom',
                       source='local', path=model_name, force_reload=True)

model = model.to(device)


ocr = easyocr.Reader(['en'], gpu=True)
elapsed = time.perf_counter() - start
print(f'Took {elapsed} seconds to load Resources.')


allowlist = string.digits + string.ascii_letters

capture_device = 1
# capture_device = 'rtsp://192.168.1.200:8080/h264_ulaw.sdp'

# frame = cv2.VideoCapture(capture_device)

# frame_width = int(frame.get(3))
# frame_height = int(frame.get(4))
# size = (frame_width, frame_height)
# writer = cv2.VideoWriter(
#     'video_output/'+str(uuid.uuid4()) + '_output.mp4', -1, 8, size)

text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0
ctr = 0
# Inference Loop


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/5
    return 1


def get_distance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


def remove_previous_files():
    files = glob.glob('./plate_capture/*')
    for f in files:
        print(f)
        os.remove(f)


def save_to_json(plate):
    if plate:
        with open('result.json') as f:
            try:
                data = json.load(f)
                if not data:
                    data = []
            except:
                data = []
            data.append(plate)
            with open('result.json', 'w') as f:
                json.dump(data, f, indent=4,
                          separators=(',', ': '))


if __name__ == '__main__':

    remove_previous_files()
    print('All temp files cleared.')

    streamer = WebcamVideoStream(capture_device)

    streamer.start()

    last_time = datetime.now()

    while True:
        ctr = ctr + 1
        # ret, image = frame.read()
        image = streamer.read()
        if image is not None:
            if (datetime.now() - last_time).microseconds/1000 > 200:
                # print(f'{(datetime.now() - last_time).microseconds=}')
                last_time = datetime.now()
                output = model(image)
                result = np.array(output.pandas().xyxy[0])
                start = time.perf_counter()
                if len(result) > 0:
                    for i in result:
                        p1 = (int(i[0]), int(i[1]))
                        p2 = (int(i[2]), int(i[3]))

                        text_origin = (int(i[0]), int(i[1])-3)

                        # drawing bounding boxes
                        cv2.rectangle(image, p1, p2, color=color, thickness=2)
                        # Extract bounding Box Content:
                        img = cv2.rectangle(
                            image, p1, p2, color=color, thickness=2)
                        # Retreive Plate number
                        plate_id = get_bbox_content(img)

                        save_to_json(plate_id)

                        font_scale = get_optimal_font_scale(
                            plate_id, get_distance(p1, (int(i[2]), int(i[1])))
                        )

                        # get boundary of this text
                        # text_size = cv2.getTextSize(
                        #     plate_id, text_font, font_scale, 2)[0]

                        # text_origin_center = (
                        #     int(i[2]-text_size[0]/2), int(i[0] + text_size[1]/2))

                        # print(text_origin_center)

                        # get coords based on boundary
                        # text_origin_center = (
                        #     int((img.shape[1] - text_size[0]) / 2), int((img.shape[0] + text_size[1]) / 2))

                        cv2.putText(
                            image,
                            text=plate_id,
                            org=text_origin,
                            fontFace=text_font,
                            fontScale=font_scale,
                            color=color,
                            thickness=2
                        )  # class and confidence text
            # Add our Company
            cv2.putText(
                image,
                text='FALCONS.AI',
                org=(7, 70),
                fontFace=text_font,
                fontScale=3,
                color=(0, 0, 255),
                thickness=3,
                lineType=cv2.LINE_AA
            )
            #cv2.putText(image, text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # writer.write(image)
            # Show image frame by frame and combine into video file
            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            text_size = cv2.getTextSize(fps, font, 3, 3)[0]

            cv2.putText(image, fps, (image.shape[1]-(text_size[0]+7), 115), font, 3,
                        (100, 255, 0), 3, cv2.LINE_AA)

            elapsed = time.perf_counter() - start
            cv2.putText(
                image,
                text=f'{round(elapsed, 2)} seconds',
                org=(7, 105),
                fontFace=text_font,
                fontScale=2,
                color=(90, 85, 68),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imshow("ANPR - FALCONS.AI", image)
            print(f'Took {elapsed} seconds to process the frame.')

        if waitKey(1) & 0xFF == ord('q'):
            break
    streamer.stop()
    cv2.destroyAllWindows()
