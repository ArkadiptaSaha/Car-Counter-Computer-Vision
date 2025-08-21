import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# ----------------------------
# Inputs
# ----------------------------
VIDEO_PATH = "cars.mp4"
MASK_PATH = "mask-950x480.png"     # White region = keep, black = ignore
#GRAPHICS_PATH = "graphics.png"     # Optional PNG with alpha for HUD overlay

# ----------------------------
# Load video
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    raise SystemExit

# Read a first frame to get dimensions
ok, first_frame = cap.read()
if not ok or first_frame is None:
    print("Error: Could not read a frame from video.")
    raise SystemExit

H, W = first_frame.shape[:2]

# ----------------------------
# Load YOLOv8 model
# ----------------------------
model = YOLO("yolov8l.pt")

# COCO class names for YOLOv8
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ----------------------------
# Prepare mask (resize once, make binary)
# ----------------------------
mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
if mask is None:
    print(f"Warning: Mask not found at '{MASK_PATH}'. Using full frame.")
    mask_bin = np.ones((H, W), dtype=np.uint8) * 255  # keep everything
else:
    # Resize mask to frame size
    mask = cv2.resize(mask, (W, H))

    # Convert to grayscale if needed
    if mask.ndim == 3:
        # If mask has alpha channel (BGRA), prefer alpha as mask if present
        if mask.shape[2] == 4:
            alpha = mask[:, :, 3]
            mask_gray = alpha
        else:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    # Binarize (assume white region should be kept)
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

# ----------------------------
# Tracking (SORT)
# ----------------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting line: [x1, y1, x2, y2]
limits = [400, 297, 673, 297]
totalCount = []

# Restart video from beginning since we consumed one frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ----------------------------
# Processing loop
# ----------------------------
while True:
    success, img = cap.read()
    if not success or img is None:
        break

    # Ensure frame matches the first frame size (optional safeguard)
    if (img.shape[1], img.shape[0]) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    # Create masked region using binary mask (single channel)
    imgRegion = cv2.bitwise_and(img, img, mask=mask_bin)


    # Run YOLO on masked region
    results = model(imgRegion, stream=True)

    # Collect detections for SORT: [x1, y1, x2, y2, conf]
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = float(box.conf[0])

            # Class
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                currentClass = classNames[cls]
            else:
                # Unknown class index, skip
                continue

            # Filter vehicles with proper precedence
            if (currentClass in ["car", "truck", "bus", "motorbike"]) and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf], dtype=float)
                detections = np.vstack((detections, currentArray))

    # Update tracker
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Track and count
    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Fancy rectangle and ID label
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(track_id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Center point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count when crossing line with ±15 px tolerance
        if limits[0] < cx < limits[2] and (limits[1] - 15) < cy < (limits[1] + 15):
            if int(track_id) not in totalCount:
                totalCount.append(int(track_id))
                # Turn line green on successful count
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Show count in top-left box
    count_text = f"Car Count = {len(totalCount)}"
    cvzone.putTextRect(img, count_text, (30, 50), scale=2, thickness=2, offset=10, colorR=(0, 0, 0))


    # Display
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)  # Uncomment to see masked input

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
