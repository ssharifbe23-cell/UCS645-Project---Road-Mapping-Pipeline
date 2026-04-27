
import cv2
import numpy as np
import os
import threading
import time
from queue import Queue

print("PARALLEL PIPELINE SYSTEM STARTED")

# ================= YOLO =================
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ================= PATH =================
img_dir = r"C:\Users\samia\OneDrive\Desktop\SE project\BDD100k"
image_files = sorted(os.listdir(img_dir))

# ================= QUEUES =================
q1 = Queue(maxsize=5)
q2 = Queue(maxsize=5)
q3 = Queue(maxsize=5)
q4 = Queue(maxsize=5)

# ================= STAGE 1: PRODUCER =================
def producer():
    print("Producer started")
    for file in image_files[:30]:
        q1.put(file)
    q1.put(None)

# ================= STAGE 2: PREPROCESS =================
def stage1_preprocess():
    print("Stage1 (Preprocess) started")

    while True:
        file = q1.get()

        if file is None:
            q2.put(None)
            break

        start = time.time()

        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness > 130:
            condition = "clear"
        elif brightness > 80:
            condition = "normal"
        else:
            condition = "low_light"
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)

        print("Stage1 time:", round(time.time() - start, 3))

        q2.put((img, file, condition))

# ================= STAGE 3: DETECTION =================
def stage2_detect():
    print("Stage2 (Detection) started")

    while True:
        data = q2.get()

        if data is None:
            q3.put(None)
            break

        start = time.time()

        img, file, condition = data
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)

                    x = int(cx - bw / 2)
                    y = int(cy - bh / 2)

                    boxes.append([x, y, bw, bh])
                    class_ids.append(class_id)

        print("Stage2 time:", round(time.time() - start, 3))

        q3.put((img, boxes, class_ids, condition))

# ================= STAGE 4: LANE =================
def stage3_lane():
    print("Stage3 (Lane Detection) started")

    while True:
        data = q3.get()

        if data is None:
            q4.put(None)
            break

        start = time.time()

        img, boxes, class_ids, condition = data

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                minLineLength=100, maxLineGap=50)

        if lines is not None:
            for line in lines[:20]:
                x1,y1,x2,y2 = line[0]
                cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

        print("Stage3 time:", round(time.time() - start, 3))

        q4.put((img, boxes, class_ids, condition))

# ================= STAGE 5: DISPLAY =================
def stage4_display():
    print("Stage4 (Display) started")

    prev = time.time()

    while True:
        data = q4.get()

        if data is None:
            break

        img, boxes, class_ids, condition = data

        for i, box in enumerate(boxes):
            x, y, w, h = box
            label = classes[class_ids[i]]

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
            cv2.putText(img, label, (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        curr = time.time()
        fps = 1/(curr-prev)
        prev = curr

        cv2.putText(img, f"FPS: {fps:.2f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(img, f"Condition: {condition}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.putText(img, "Pipeline: Preprocess->Detect->Lane->Output", 
            (20,120), cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (255,255,0), 2)
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2

        cv2.imshow("PARALLEL PIPELINE SYSTEM", img)

        key = cv2.waitKey(6000)  # demo delay

        if key == 27:
            break

    cv2.destroyAllWindows()

# ================= SEQUENTIAL VERSION =================
def run_sequential():
    print("\nRunning Sequential Version...")

    start = time.time()

    for file in image_files[:40]:
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

    print("Sequential Time:", round(time.time() - start, 3), "sec")

# ================= MAIN =================
if __name__ == "__main__":

    run_sequential()

    print("\nRunning Parallel Pipeline...\n")

    start_parallel = time.time()

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=stage1_preprocess)
    t3 = threading.Thread(target=stage2_detect)
    t4 = threading.Thread(target=stage3_lane)
    t5 = threading.Thread(target=stage4_display)

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    print("Parallel Time:", round(time.time() - start_parallel, 3), "sec")