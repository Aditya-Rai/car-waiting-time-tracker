SOURCE = "input_videos/video2.mp4"                  # Input video_path
MODEL_PATH = "models/yolo11s.pt"                    # Input model path 
MODEL_CONF = 0.25                                   # Model confidence
MODEL_IOU = 0.7                                     # Model IOU threshold
MODEL_IMGSZ = 928                                   # Model image size


QUEUE_SIZE = 2                                      # Queue Size used in the code
MISSING_THRESH = 100                                # Threshold to remove the tracked car, when disappears for n frames
LOG_LEVEL="DEBUG"                                   # Log Level :TODO : Use this log level
# Default ROI that will be considered during inference
DEFAULT_ROI= [(0.13958333333333334, 0.48518518518518516), (0.43854166666666666, 0.46574074074074073), (0.5885416666666666, 0.9833333333333333), (0.04479166666666667, 0.9935185185185185)]
SHOW_INFERENCE=True                                 # Show the inference , keep True
SAVE_INFERENCE=True                                 # Save inference , True or False
SAVE_VIDEO_PATH = "output/output.avi"
SAVE_INFERENCE_SIZE = (1280, 720)