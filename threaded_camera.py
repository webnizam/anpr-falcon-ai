from shutil import ExecError
from termios import ECHOE
import cv2
from threading import Thread
from typing import Any


class ThreadedCamera(object):
    def __init__(self, source: Any = 0):
        try:
            self.capture = cv2.VideoCapture(source)
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            self.status = False
            self.frame = None
        except Exception as e:
            raise Exception(e)

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None
