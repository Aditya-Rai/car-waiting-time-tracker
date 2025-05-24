
from utils.utils import get_waiting_time

class CarObject:
    def __init__(self,track_id,present_count,missing_count,norm_bbox,denorm_bbox):
        self.track_id = track_id
        self.present_count = present_count
        self.missing_count = missing_count
        self.waiting_time = get_waiting_time(self.present_count)
        self.norm_bbox = norm_bbox
        self.denorm_bbox = denorm_bbox

    def increase_present_count(self):
        self.present_count += 1
    
    def increase_missing_count(self):
        self.missing_count += 1
    
    def reset_missing_count(self):
        self.missing_count = 0

    def update(self,norm_bbox,denorm_bbox):
        self.norm_bbox = norm_bbox
        self.denorm_bbox = denorm_bbox
        

    

        