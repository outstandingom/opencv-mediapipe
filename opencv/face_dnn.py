import cv2
import os

proto = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, model)

video = "/storage/emulated/0/Download/test_fixed.mp4"
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("❌ Video not found")
    exit()

print("▶ Detecting faces using OpenCV DNN...")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1

    if face_count > 0:
        print(f"Frame {frame_id}: {face_count} face(s)")

cap.release()
print("✅ Done.")
