import cv2
from threading import Thread
from typing import Any


class ThreadedCamera(object):
    def __init__(self, source: Any = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                if not self.status:
                    cv2.destroyAllWindows()
                    break

    def grab_frame(self):
        if self.status:
            return self.frame
        return None
