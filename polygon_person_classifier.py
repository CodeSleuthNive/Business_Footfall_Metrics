import cv2
import numpy as np
import json
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load ROI points from file
with open("roi_polygon.json", "r") as f:
    points = json.load(f)
EMPLOYEE_REGION_POINTS = np.array(points, np.int32).reshape((-1, 1, 2))

def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Load video
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# Get video info
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    employee_count = 0
    customer_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:  # person class
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if is_inside_polygon(cx, cy, EMPLOYEE_REGION_POINTS):
                    label = "Employee"
                    color = (0, 255, 0)
                    employee_count += 1
                else:
                    label = "Customer"
                    color = (255, 0, 0)
                    customer_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw polygon
    cv2.polylines(frame, [EMPLOYEE_REGION_POINTS], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show counts
    cv2.putText(frame, f"Employees: {employee_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Customers: {customer_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
