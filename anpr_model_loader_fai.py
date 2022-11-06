import torch
import numpy as np


#load anpr model 
#credits to https://www.falcons.ai
class FaiAnprModelLoader:
    def __init__(self, path, driver):
        self.path = path
        self.device = torch.device(driver)
        self.model = torch.hub.load(
            'ultralytics-yolov5-6371de8/',
            'custom',
            source='local',
            path=path,
            force_reload=True
        )
        self.model = self.model.to(self.device)

    def get_number_plates(self, image):
        output = self.model(image)
        return np.array(output.pandas().xyxy[0])
