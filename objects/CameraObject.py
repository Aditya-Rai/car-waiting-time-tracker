import os
import time
from threading import Thread
from queue import Queue
import json
import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

from custom_tracker.tracker import BYTETracker as ByteTracker
from objects.CarObject import CarObject
from utils.utils import create_logger, get_iou, get_waiting_time


class CameraObject:
    def __init__(self):
        # Initialize logger for this class
        self.logger = create_logger("CameraObject", 1)

        # Load configuration from JSON file
        self.config_file = "config.json"
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config file '{self.config_file}': {e}")
            raise

        # Assign config values to instance variables
        self.source = config.get("SOURCE")
        self.model_path = config.get("MODEL_PATH")
        self.model_conf = config.get("MODEL_CONF")
        self.model_iou = config.get("MODEL_IOU")
        self.model_imgsz = config.get("MODEL_IMGSZ")
        self.queue_size = config.get("QUEUE_SIZE")
        self.missing_thresh = config.get("MISSING_THRESH")
        self.log_level = config.get("LOG_LEVEL")
        self.default_roi = config.get("DEFAULT_ROI")
        self.show_inference = config.get("SHOW_INFERENCE")
        self.save_inference = config.get("SAVE_INFERENCE")
        self.save_video_path = config.get("SAVE_VIDEO_PATH")
        self.save_inference_size = tuple(config.get("SAVE_INFERENCE_SIZE"))

        self.logger.info(f"Initializing CameraObject with source: {self.source}")

        # Attempt to open the video capture source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open the source: {self.source}")
            print(f"Cannot open Source path defined: {self.source}")
            os._exit(0)
        else:
            # Retrieve camera properties
            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Camera initialized with width: {self.width}, height: {self.height}, fps: {self.fps}")

            # If video saving is enabled, set up the video writer
            if self.save_inference:
                file_name = os.path.basename(self.save_video_path)
                dir_name = os.path.dirname(self.save_video_path)

                self.logger.info(f"Configured video output file name: {file_name}")
                self.logger.info(f"Configured video output directory: {dir_name}")

                if not file_name:
                    self.logger.warning("File name not found, defaulting to output.avi")

                os.makedirs(dir_name, exist_ok=True)
                save_path = os.path.join(dir_name, file_name)
                self.logger.info(f"Saving the output video at: {save_path}")

                self.video_writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    self.fps,
                    self.save_inference_size
                )

        # Initialize Region of Interest (ROI)
        self.denorm_roi = self.get_default_roi()
        self.roi = self.default_roi
        self.logger.info("Starting with the default ROI")
        self.logger.info("Create a Custom ROI by clicking on the inference screen")

        # Initialize queues for multi-threaded processing
        self.logger.info("Initializing internal queues")
        self.reader_queue = Queue(maxsize=1)
        self.detection_queue = Queue(maxsize=self.queue_size)
        self.processing_queue = Queue(maxsize=self.queue_size)
        self.writer_queue = Queue(maxsize=self.queue_size)

        # Load the YOLO detection model
        self.logger.info("LOADING the car detection model")
        self.det_model = YOLO(self.model_path, task="detect")
        self.logger.info("LOADED the car detection model")

        # Load the custom ByteTrack tracker
        self.logger.info("LOADING the custom tracker")
        self.tracker = ByteTracker(30, 0.01, 0.005, 0.9, 50, 640, False)
        self.logger.info("LOADED the custom tracker")

        # Initialize runtime control and data containers
        self.is_running = True
        self.tracked_data = {}   # For storing current tracked objects
        self.old_data = {}       # For storing previously tracked data

        self.temp_roi = []       # Temporary ROI drawn by the user
        self.inference_frame = None  # Current frame for inference display

    def save_config(self):
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config for saving: {e}")
            config = {}

        # Update the ROI points with current normalized ROI
        config["DEFAULT_ROI"] = self.roi

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"Saved updated ROI to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save updated ROI: {e}")

    def get_in_range_polygon(self, roi_points):
        """
        Clamp ROI point coordinates to the range [0, 1].

        This ensures normalized ROI points do not exceed the image bounds when denormalized.
        """
        self.logger.debug("Clamping ROI points to the [0, 1] range.")
        for i, point in enumerate(roi_points):
            clamped_x = max(0, min(1, point[0]))
            clamped_y = max(0, min(1, point[1]))
            roi_points[i] = [clamped_x, clamped_y]
            self.logger.debug(f"Point {i}: Original={point}, Clamped={[clamped_x, clamped_y]}")
        return roi_points


    def denormalize_roi_polygon(self, roi_points_original, H, W):
        """
        Convert normalized ROI points to pixel coordinates using the frame's height and width.

        Args:
            roi_points_original (list): Normalized ROI points.
            H (int): Image height.
            W (int): Image width.

        Returns:
            list: Denormalized ROI points in pixel format.
        """
        self.logger.debug(f"Denormalizing ROI polygon with frame size: H={H}, W={W}")
        roi_points = roi_points_original.copy()

        # Clamp to valid range before scaling
        roi_points = self.get_in_range_polygon(roi_points)

        for i, point in enumerate(roi_points):
            x_pixel = int(point[0] * W)
            y_pixel = int(point[1] * H)
            roi_points[i] = [x_pixel, y_pixel]
            self.logger.debug(f"Point {i}: Normalized={point}, Pixel={[x_pixel, y_pixel]}")
        
        return roi_points


    def get_camera_roi(self, rois):
        """
        Converts ROI points into a format usable by OpenCV drawing and masking.

        Args:
            rois (list): Normalized ROI point list.

        Returns:
            np.ndarray: Denormalized and reshaped polygon for OpenCV, or None if invalid input.
        """
        if rois is None or rois == []:
            self.logger.warning("Empty or None ROI received. Returning None.")
            return None

        self.logger.info("Generating camera ROI polygon from normalized points.")
        H = self.height
        W = self.width

        denorm_roi = self.denormalize_roi_polygon(rois, H, W)

        # Create polygon using shapely and convert to OpenCV format
        polygon = Polygon(denorm_roi)
        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        coords = coords.reshape((-1, 1, 2))
        
        self.logger.debug(f"Final polygon shape: {coords.shape}")
        return coords


    def get_default_roi(self):
        """
        Get the default region of interest polygon using the DEFAULT_ROI constant.

        Returns:
            np.ndarray: Default ROI polygon for drawing/inference use.
        """
        self.logger.info("Retrieving default ROI polygon.")
        default_roi = self.default_roi
        denorm_rois = self.get_camera_roi(default_roi)
        return denorm_rois

    def preprocess_frame(self,frame):
        """
        This function preprocesses the frame, draws the removes the part of the frame that is outside of the ROI
        """
        if self.denorm_roi is not None:
            blank_dummy_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            blank_dummy_image = cv2.fillPoly(blank_dummy_image, [self.denorm_roi], (255, 255, 255))
            blank_dummy_image = cv2.bitwise_and(frame, blank_dummy_image)
            frame = blank_dummy_image
        return frame

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            normalize_point = (x / self.inference_frame.shape[1], y / self.inference_frame.shape[0])
            self.temp_roi.append(normalize_point)
            # self.logger.info(f"Temp ROI: {self.temp_roi}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.temp_roi) < 2:
                return
            self.denorm_roi = self.get_camera_roi(self.temp_roi)
            self.roi = self.temp_roi
            self.logger.info(f"ROI: {self.roi}")
            self.temp_roi = []

            # Save the updated normalized ROI to config.json
            self.save_config()

    
    def reader(self):
        """
        This function reads the frames from the camera feed / video and puts it in the reader queue
        """
        print("Staring the reader thread")
        self.logger.info(f"Starting the reader thread")
        ret_false_counter = 0
        while self.is_running:
            try:
                ret,frame = self.cap.read()
                if not ret:
                    ret_false_counter += 1
                    if ret_false_counter > 10:
                        print("Trying to reconnect, Camera feed not available")
                        self.logger.error(f"Camera feed not available ,trying to reconnect")
                        self.cap.release()
                        self.stop()
                        self.cap = cv2.VideoCapture(self.source)
                        self.reader_queue.queue.clear()
                        time.sleep(2)
                        continue
                    else:
                        time.sleep(0.01)
                        continue
                else:
                    ret_false_counter = 0
                    while self.reader_queue.full():
                        time.sleep(0.01)
                        continue
                    
                    data = {"frame":frame,"frame_start_time":time.time()}       # putting the data inside the reader queue
                    self.reader_queue.put(data)   
                    
            except Exception as e:
                self.logger.error(f"Error in reader thread {e}",exc_info=True)
                self.camera_reader.release()
                self.stop()
        print("Camera feed stopped")
        self.logger.info(f"Camera feed stopped")
            
    def detector(self):
        print(f"Staring the detection thread")
        self.logger.info(f"Starting the detection thread")
        while self.is_running:
            if self.reader_queue.empty():
                time.sleep(0.01)
                continue
            data = self.reader_queue.get()
            frame = data["frame"]
            frame_preprocess = self.preprocess_frame(frame.copy())

            print(f"reading frame")
            detections = {}
            try:
                start_time = time.time()
                # results = self.det_model.predict(frame.copy(),imgsz=MODEL_IMGSZ,conf=MODEL_CONF, iou=MODEL_IOU,verbose=False)[0]
                # print(results)
                results = self.det_model.predict(frame_preprocess.copy(),imgsz=self.model_imgsz,conf=self.model_conf, iou=self.model_iou,classes=[2],verbose=False)[0]
                print(f"Detection time : {time.time() - start_time}")
                norm_bboxes = results.boxes.xyxyn.detach().cpu().numpy()
                
                denorm_bboxes = results.boxes.data.detach().cpu().numpy()
                # print(f"OG : {denorm_bboxes}\n\n{norm_bboxes}")
                for box in denorm_bboxes:
                    cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
                # print(f"Frame shape : {frame.shape}")
                detections = {
                    "norm_bbox":norm_bboxes,
                    "denorm_bbox":denorm_bboxes,
                }
                
                start_time = time.time()
                # tracking based on the detections
                track_list  = self.tracker.update(denorm_bboxes)
                print(f"Tracking time : {time.time() - start_time}")
                id_list = [t.track_id for t in track_list]  # Get id list
                box_list = [t.tlbr for t in track_list]     # Get box list
                conf_list = [t.score for t in track_list]   # Get conf scores
                temp_current_frame_data = dict()
                map_start_time = time.time()
                # logic to map the bbox, and avoid skipping
                for i,tracker_bbox in enumerate(box_list):
                    track_id = id_list[i]
                    max_iou = 0
                    matched_index = None
                    for det_index,det in enumerate(denorm_bboxes):
                        iou_ = get_iou(det,tracker_bbox)
                        if iou_ > max_iou:
                            max_iou = iou_
                            matched_index = det_index
                    # print(f"Matched {tracker_bbox} to {denorm_bboxes[matched_index]} with iou {max_iou}")
                    if matched_index is not None:# and matched_index not in matched_index_dict.keys():# or matched_index_dict[matched_index]["max_iou"] < max_iou):
                        temp_current_frame_data[i] = {"track_id":track_id,"max_iou":max_iou,"det":denorm_bboxes[matched_index],"norm_det":norm_bboxes[matched_index],"missing_count":0}
                    else:
                        temp_current_frame_data[i] = {"track_id":track_id,"max_iou":None,"det":tracker_bbox,"norm_det":norm_bboxes[i],"missing_count":0}
                        
                current_frame_data = {}
                for key in temp_current_frame_data.keys():
                    track_id = temp_current_frame_data[key]["track_id"]
                    current_frame_data[track_id] = temp_current_frame_data[key]
                print(f"Matching time : {time.time() - start_time}")
                data["current_frame_data"] = current_frame_data
                
                while self.processing_queue.full():
                    time.sleep(0.01)
                    continue
                self.processing_queue.put(data)
            except Exception as e:
                self.logger.error(f"Error in detector thread : {e}",exc_info=True)
                self.stop()
        
    def postprocess(self):
        print(f"Staring the Postprocess thread")
        self.logger.info(f"Starting the Postprocess thread")
        while self.is_running:
            if self.processing_queue.empty():
                time.sleep(0.01)
                continue
            try:
                data = self.processing_queue.get()
                current_frame_data = data["current_frame_data"]
                
                tracked_copy = self.tracked_data.copy()
                old_copy = self.old_data.copy()
                self.tracked_data = {}
                self.old_data = {}
                
                for key in tracked_copy.keys():
                    current_car_data = tracked_copy[key]
                    if key in current_frame_data.keys():
                        # self.logger.info(f"Updating data for {key}")
                        current_data = current_frame_data[key]
                        current_car_data.increase_present_count()
                        current_car_data.reset_missing_count()
                        current_car_data.update(current_data["norm_det"],current_data["det"])
                        self.tracked_data[key] = current_car_data
                        del current_frame_data[key]
                    else:   
                        current_car_data.increase_missing_count()
                        self.old_data[key] = current_car_data

                for key in old_copy.keys():
                    current_car_data = old_copy[key]
                    if key not in current_frame_data.keys():
                        if current_car_data.missing_count < self.missing_thresh:
                            current_car_data.increase_missing_count()
                            self.old_data[key] = current_car_data
                        else:
                            self.logger.info(f"Deleted id {key}")
                    else:   
                        current_car_data.present_count = current_car_data.present_count + current_car_data.missing_count + 1
                        current_car_data.reset_missing_count()
                        self.tracked_data[key] = current_car_data
                        del current_frame_data[key]
                
                for key in current_frame_data.keys():
                    # self.logger.info(f"Creating new data for {key}")
                    current_data = current_frame_data[key]
                    car_object = CarObject(current_data["track_id"],0,0,current_data["norm_det"],current_data["det"])
                    self.tracked_data[key] = car_object
                    
                    
                data["tracked_data"] = self.tracked_data
                data["old_data"] = self.old_data
                
                while self.writer_queue.full():
                    time.sleep(0.01)
                    continue
                self.writer_queue.put(data)
                    
            except Exception as e:
                self.logger.error(f"Error in postprocess thread : {e}",exc_info=True)
                self.stop()
        
        
    def writer_thread(self):
        print(f"Staring the Writer thread")
        self.logger.info(f"Starting the Writer thread")
        while self.is_running:
            if self.writer_queue.empty():
                time.sleep(0.01)
                continue
            start_time = time.time()
            try:
                data = self.writer_queue.get()
                frame = data["frame"]
                tracked_data = data["tracked_data"]
                for key in tracked_data.keys():
                    current_car = tracked_data[key]
                    present_count = current_car.present_count
                    seconds = round((present_count / self.fps), 2)
                    norm_bbox = current_car.norm_bbox
                    track_id = current_car.track_id
                    denorm_bbox = [
                        norm_bbox[0]*frame.shape[1],
                        norm_bbox[1]*frame.shape[0],
                        norm_bbox[2]*frame.shape[1],
                        norm_bbox[3]*frame.shape[0]
                    ]
                    x1, y1, x2, y2 = map(int, denorm_bbox[:4])
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Convert time to hh:mm:ss
                    watiting_time = get_waiting_time(seconds)
                    text = f"{track_id}|{watiting_time}"

                    # Dynamically determine the font scale so text fits within the bounding box width
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    text_thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)

                    while text_size[0] > box_width - 4 and font_scale > 0.1:
                        font_scale -= 0.05
                        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)

                    text_width, text_height = text_size
                    bg_height = int(text_height + 10)


                    # Draw filled background rectangle for text

                    # Calculate centered text position
                    text_x = x1 + (box_width - text_width) // 2
                    text_y = y1 + bg_height - 5
                    text_position = (text_x, text_y)

                    # Put text
                    cv2.rectangle(frame, (x1, y1), (x2, y1 + bg_height), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 0), text_thickness)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)


                print(f"Display Time : {time.time()-start_time}")
                self.inference_frame = frame
                cv2.namedWindow("Inference Window", flags=cv2.WINDOW_GUI_NORMAL)
                cv2.setMouseCallback("Inference Window",self.click_event)
                denormalized_points = []
         
                # print(frame.shape)
                for point in self.temp_roi:
                    denormalized_points.append((int(point[0]*self.inference_frame.shape[1]),int(point[1]*self.inference_frame.shape[0])))
                cv2.polylines(self.inference_frame,[np.array(denormalized_points)],True,(0,255,255),2)
            
                denormalized_points = []
                for point in self.roi:
                    denormalized_points.append((int(point[0]*self.inference_frame.shape[1]),int(point[1]*self.inference_frame.shape[0])))
                cv2.polylines(self.inference_frame,[np.array(denormalized_points)],True,(0,255,0),2)
                

                writer_frame = cv2.resize(self.inference_frame,self.save_inference_size)

                if self.save_inference:
                    self.video_writer.write(writer_frame)
                if self.show_inference:
                    cv2.imshow("Inference Window",self.inference_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.stop()
                    break
            except Exception as e:
                self.logger.error(f"Error in writer thread : {e}",exc_info=True)
                self.stop()
        
        
    def start(self):
        """
        This funciton is responsible for starting all the threads
        """
        self.all_threads = []
        t = Thread(target=self.reader, daemon=True)
        self.all_threads.append(t)
        t = Thread(target=self.detector, daemon=True)
        self.all_threads.append(t)
        t = Thread(target=self.postprocess, daemon=True)
        self.all_threads.append(t)
        t = Thread(target=self.writer_thread, daemon=True)
        self.all_threads.append(t)
        while self.is_running:
            for thread in self.all_threads:
                if not thread.is_alive():
                    thread.start()
                    self.logger.info(f"Thread {thread.name} started")
            time.sleep(1)
            
    
    def stop(self):
        self.is_running = False
        time.sleep(5)
        for thread in self.all_threads:
            thread.join()
            self.logger.info(f"Thread {thread.name} stopped")
        self.video_writer.release()
            
            
            
    
            
        



