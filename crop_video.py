import os
import cv2
import numpy as np

input_video_path = "input_videos/video1.mp4"
output_video_path = "output_videos/video3.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(input_video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
print("width: " + str(width))
print("height: " + str(height))
print("fps: " + str(fps))

crop_start_time = {
    "hour":0,
    "minute":1,
    "second":9
}

crop_end_time = {
    "hour":0,
    "minute":1,
    "second":12
    
    
    # "minute":2,
    # "second":33
}

frame_counter = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret:
        frame_counter += 1
        if frame_counter < (crop_start_time["hour"] * 60 * 60 + crop_start_time["minute"] * 60 + crop_start_time["second"])*fps:
            continue
        if frame_counter > (crop_end_time["hour"] * 60 * 60 + crop_end_time["minute"] * 60 + crop_end_time["second"])*fps:
            break
        
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.write(frame)
    else:
        break

cap.release()
cv2.destroyAllWindows()
writer.release()