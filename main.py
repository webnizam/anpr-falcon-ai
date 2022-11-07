# Falcons.ai
# Michael Stattelman 2022
import glob
import os
import sys
from anpr_model_loader_fai import FaiAnprModelLoader
import cv2
import time
import string
from datetime import datetime
#from pytesseract import image_to_string
# from paddleocr import PaddleOCR,draw_ocr
import easyocr
import time
import json
import argparse
from threaded_camera import WebcamVideoStream

allowlist = string.digits + string.ascii_letters

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-d',
    '--device',
    type=str,
    default=0,
    help='Choose the device',
)

args = parser.parse_args()

capture_device = int(args.device) if str(args.device).isdigit() else str(
    args.device)

text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0
ctr = 0


def get_Text(result):
    try:
        txts = [line[1] for line in result if len(line[1]) < 6]

        if len(txts) > 4:
            txts = txts[:3]

        return ' '.join(txts)
    except Exception as e:
        print(e)
        return ""


def resize_image(image, height=400):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    window_width = height / aspect_ratio
    image = cv2.resize(image, (int(height), int(window_width)),
                       interpolation=cv2.INTER_LANCZOS4)
    return image


def remove_img(img_name):
    os.remove(img_name)


# Extract the image from the bounding box
def get_bbox_content(img):
    result = ocr.readtext(resize_image(img), allowlist=allowlist)
    plate_num = get_Text(result)
    return plate_num


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text,
                                   fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                   fontScale=scale / 10,
                                   thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 5
    return 1


def get_distance(p1, p2):
    dis = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
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
        data = list(set(data))
        with open('result.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))


def put_auth_stat(image, authorized):
    height_, width_, c = image.shape

    if authorized:
        text_ = 'Authorized'
        text_size_ = cv2.getTextSize(text_, text_font, 5, 2)[0]
        origin_ = (int((width_ - text_size_[0]) / 2), height_ - text_size_[1])
        cv2.putText(image,
                    text=text_,
                    org=origin_,
                    fontFace=text_font,
                    fontScale=5,
                    color=(0, 255, 0),
                    thickness=3)
    else:
        text_ = 'Not Authorized'
        text_size_ = cv2.getTextSize(text_, text_font, 5, 2)[0]
        origin_ = (int((width_ - text_size_[0]) / 2), height_ - text_size_[1])
        cv2.putText(image,
                    text=text_,
                    org=origin_,
                    fontFace=text_font,
                    fontScale=5,
                    color=(0, 0, 255),
                    thickness=3)


if __name__ == '__main__':
    start = time.perf_counter()
    model_path = r"best.pt"
    cpu_or_cuda = "mps" if sys.platform == 'darwin' else 'cuda'
    anpr = FaiAnprModelLoader(model_path, cpu_or_cuda)
    ocr = easyocr.Reader(['en'], gpu=True)
    elapsed = time.perf_counter() - start
    print(f'Took {elapsed} seconds to load Resources.')

    remove_previous_files()
    print('All temp files cleared.')

    with open('authorized_plates.json') as f:
        try:
            authorized_plates = json.load(f)
            if not authorized_plates:
                authorized_plates = []
        except:
            authorized_plates = []

    print(f'{authorized_plates=}')
    try:
        streamer = WebcamVideoStream(capture_device)
    except Exception as e:
        print('Its the error', e, sep='\n\n')
        exit(0)

    streamer.start()

    last_time = datetime.now()

    last_auth_time = None
    last_auth_read_count = 0
    authorized = False
    auth_timeout_seconds = 5
    auth_read_threshold = 5

    while True:
        ctr = ctr + 1
        # ret, image = frame.read()
        image = streamer.read()
        do_process = (datetime.now() - last_time).microseconds / 1000 > 100
        # do_process = True
        if image is not None:
            if do_process:
                # print(f'{(datetime.now() - last_time).microseconds=}')
                last_time = datetime.now()
                # output = model(image)
                # result = np.array(output.pandas().xyxy[0])

                result = anpr.get_number_plates(image)
                start = time.perf_counter()
                if len(result) > 0:
                    for i in result:
                        p1 = (int(i[0]), int(i[1]))
                        p2 = (int(i[2]), int(i[3]))

                        text_origin = (int(i[0]), int(i[1]) - 3)

                        # drawing bounding boxes
                        cv2.rectangle(image, p1, p2, color=color, thickness=2)
                        # Extract bounding Box Content:
                        img = cv2.rectangle(image,
                                            p1,
                                            p2,
                                            color=color,
                                            thickness=2)
                        # Retreive Plate number
                        plate_id = get_bbox_content(img)

                        save_to_json(plate_id)

                        font_scale = get_optimal_font_scale(
                            plate_id, get_distance(p1, (int(i[2]), int(i[1]))))

                        cv2.putText(image,
                                    text=plate_id,
                                    org=text_origin,
                                    fontFace=text_font,
                                    fontScale=font_scale,
                                    color=color,
                                    thickness=2)
                        for plate in authorized_plates:
                            if plate in plate_id:
                                authorized = True
                                last_auth_time = datetime.now()
                                last_auth_read_count += 1
                                # print(f'{last_auth_read_count=}')
                                break

            if authorized and last_auth_time and (datetime.now() - last_auth_time).total_seconds() < auth_timeout_seconds \
                    and last_auth_read_count > auth_read_threshold:
                put_auth_stat(image, True)
            else:
                put_auth_stat(image, False)

            if last_auth_time and (datetime.now() - last_auth_time
                                   ).total_seconds() > auth_timeout_seconds:
                last_auth_read_count = 0
                authorized = False

            # Add our Company
            cv2.putText(image,
                        text='FALCONS.AI',
                        org=(7, 70),
                        fontFace=text_font,
                        fontScale=3,
                        color=(0, 0, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA)
            #cv2.putText(image, text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # writer.write(image)
            # Show image frame by frame and combine into video file
            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            text_size = cv2.getTextSize(fps, font, 3, 3)[0]

            cv2.putText(image, fps, (image.shape[1] - (text_size[0] + 7), 115),
                        font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            elapsed = time.perf_counter() - start
            cv2.putText(image,
                        text=f'{round(elapsed, 2)} seconds',
                        org=(7, 105),
                        fontFace=text_font,
                        fontScale=2,
                        color=(90, 85, 68),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.imshow("ANPR - FALCONS.AI", image)
            print(f'Took {elapsed} seconds to process the frame.')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.stop()
    cv2.destroyAllWindows()
