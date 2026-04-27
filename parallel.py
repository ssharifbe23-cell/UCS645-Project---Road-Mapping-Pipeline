import cv2
import numpy as np
import os
import time
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt

# ================= QUEUES =================
q1 = Queue(maxsize=5)
q2 = Queue(maxsize=5)
q3 = Queue(maxsize=5)
q4 = Queue(maxsize=5)

# ================= GLOBALS =================
vehicle_history = []
prev_centers = []

# ================= DATA =================
img_dir = "BDD100k"
image_files = sorted(os.listdir(img_dir))[:20]

# ================= MODEL =================
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# ================= STAGE 1 =================
def reader():
    for file in image_files:
        img = cv2.imread(os.path.join(img_dir, file))
        if img is not None:
            q1.put(img)
    q1.put(None)

# ================= STAGE 2 =================
def preprocess():
    while True:
        img = q1.get()
        if img is None:
            q2.put(None)
            break

        enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        q2.put(enhanced)

# ================= STAGE 3 =================
def detect():
    while True:
        img = q2.get()
        if img is None:
            q3.put(None)
            break

        blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(layer_names)

        q3.put((img, outputs))

# ================= STAGE 4 =================
def analyze():
    global prev_centers

    while True:
        data = q3.get()
        if data is None:
            q4.put(None)
            break

        img, outputs = data
        h, w = img.shape[:2]

        centers = []
        vehicle_count = 0

        for out in outputs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > 0.5 and class_id == 2:  # car
                    cx = int(det[0] * w)
                    cy = int(det[1] * h)
                    centers.append((cx, cy))
                    vehicle_count += 1

        # FLOW ESTIMATION
        direction = "STABLE"
        if prev_centers and centers:
            avg_prev = np.mean(prev_centers, axis=0)
            avg_curr = np.mean(centers, axis=0)

            if avg_curr[0] > avg_prev[0]:
                direction = "RIGHT"
            elif avg_curr[0] < avg_prev[0]:
                direction = "LEFT"

        prev_centers = centers

        q4.put((img, vehicle_count, direction))

# ================= STAGE 5 (UPGRADED) =================
def display():
    frame_count = 0
    start_time = time.time()

    plt.ion()

    while True:
        data = q4.get()
        if data is None:
            break

        img, count, direction = data
        original = img.copy()

        # PERFORMANCE
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        # ALERT SYSTEM
        if count > 10:
            alert = "HIGH TRAFFIC"
            color = (0,0,255)
        else:
            alert = "NORMAL"
            color = (0,255,0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < 80:
            visibility = "LOW VISIBILITY"
        else:
            visibility = "GOOD VISIBILITY"

        # TEXT OVERLAY
        cv2.putText(img, f"Vehicles: {count}", (20,40), 0, 0.7, (0,255,0),2)
        cv2.putText(img, f"Flow: {direction}", (20,80), 0, 0.7, (0,255,0),2)
        cv2.putText(img, f"Throughput: {fps:.2f}", (20,120), 0, 0.7, (0,255,0),2)

        cv2.putText(img, f"Traffic: {alert}", (20,160), 0, 0.8, color,2)
        cv2.putText(img, f"Visibility: {visibility}", (20,200), 0, 0.7, (255,255,0),2)

        # SIDE-BY-SIDE VIEW
        combined = np.hstack((original, img))
        combined = cv2.resize(combined, (1000, 500))

        cv2.imshow("Parallel Intelligent Traffic System", combined)

        # TRAFFIC TREND GRAPH
        vehicle_history.append(count)

        if len(vehicle_history) > 50:
            vehicle_history.pop(0)

        plt.clf()
        plt.plot(vehicle_history)
        plt.title("Traffic Density Over Time")
        plt.xlabel("Frames")
        plt.ylabel("Vehicles")
        plt.pause(0.001)

        # EXIT
        if cv2.waitKey(800) == 27:
            break

    cv2.destroyAllWindows()

# ================= RUN =================
threads = [
    Thread(target=reader),
    Thread(target=preprocess),
    Thread(target=detect),
    Thread(target=analyze),
    Thread(target=display)
]

for t in threads:
    t.start()

for t in threads:
    t.join()