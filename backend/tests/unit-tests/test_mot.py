from velour_api.mot_metrics import compute_mot_metrics
import schemas

def generate_mot_data(num_frames: int, num_dets_per_frame: int, prediction: bool=True):
    create_img = lambda frame: schemas.Image(uri="",height=500,width=500,frame=frame)
    create_
