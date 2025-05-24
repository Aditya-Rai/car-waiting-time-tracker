# 🚗 Vehicle Wait Time Tracker in ROI

This project is a computer vision application to detect, track, and calculate the wait time of vehicles within a user-defined Region of Interest (ROI) in a video. It outputs a processed video with bounding boxes, ROI display, and per-vehicle wait timers.

---

## 🚀 How to Run

### Step-by-step

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```
2. **Change the paramters in the constant.py file as required**
```bash
python main.py --logs logs
```

---
🧠 Features
---

YOLOv11 vehicle detection (custom model support) with same architecture

ByteTrack for robust multi-object tracking

Wait time tracking for each vehicle inside ROI

Real-time overlay of bounding boxes, ROI, and wait time (MM:SS)

Flexible ROI setup: default or interactive

Logging support (--logs logs)

Highly configurable via constants.py

---
✏️ How to Draw the ROI
---

By default the system will use the ROI from the constants.py

A pop-up window will open when the script starts

Maximize the window

Use Left Click to draw polygon points (one click per vertex)

Use Right Click to close and finalize the ROI polygon

Now we will have the inference based on the ROI drawn
