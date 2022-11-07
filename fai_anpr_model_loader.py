import torch
import numpy as np


#Load ANPR Model 
#Credits to https://www.falcons.ai | Michael Stattelman
class FaiAnprModelLoader:
    def __init__(self, path, driver):
        self.path = path
        self.device = torch.device(driver)
        self.model = torch.hub.load(
            'ultralytics-yolov5/',
            'custom',
            source='local',
            path=path,
            force_reload=True
        )
        self.model = self.model.to(self.device)

    def get_number_plates(self, image):
        output = self.model(image)
        return np.array(output.pandas().xyxy[0])
