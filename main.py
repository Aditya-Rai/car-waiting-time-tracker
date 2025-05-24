import argparse
from objects.CameraObject import CameraObject
import os
import warnings
import shutil
from threading import Thread
from flask import Flask, Response, render_template_string,render_template
import cv2
import time
from queue import Queue
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)
streamer_queue = Queue(maxsize=1)  # Global Queue to share frames


def check_arguments():
    parser = argparse.ArgumentParser(description='Processing logs.')
    parser.add_argument('--logs', type=str, help='The path to the logs directory')
    args = parser.parse_args()

    if args.logs:
        log_path = args.logs
        if os.path.exists(log_path) and os.path.isdir(log_path):
            print(f'Log path {log_path} exists and is a directory. Creating Backup...')

            logs_backup_path = log_path + '_backup'
            if os.path.exists(logs_backup_path):
                print(f'Log backup path {logs_backup_path} exists. Deleting...')
                shutil.rmtree(logs_backup_path)

            shutil.copytree(log_path, logs_backup_path)
            shutil.rmtree(log_path)

            print(f'Log path {log_path} deleted.')
        else:
            print(f'Log path {log_path} does not exist or is not a directory.')
    else:
        print('No log path provided.')


# Basic HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Frame Stream</title>
</head>
<body>
    <h1>Live Frame Stream</h1>
    <img src="/video_feed" width="800" />
</body>
</html>
"""


@app.route('/')
def index():
    return render_template("stream.html")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    while True:
        if streamer_queue.empty():
            time.sleep(0.01)
            continue
        frame = streamer_queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def start_web_server(port):
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    check_arguments()

    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {"PORT": 5000}

    camera_object = CameraObject(streamer_queue)

    # Start camera and web server in parallel
    Thread(target=camera_object.start, daemon=True).start()
    Thread(target=start_web_server, args=(config.get("PORT", 5000),), daemon=True).start()

    # Keep main thread alive
    while True:
        if not camera_object.is_running:
            break
        time.sleep(1)
