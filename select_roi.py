import cv2
import numpy as np
import json

# Global variables
points = []
frame = None

def select_points(event, x, y, flags, param):
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        temp_frame = frame.copy()
        for pt in points:
            cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
        cv2.imshow("Select ROI", temp_frame)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2:
            polygon = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Select ROI", frame)
            cv2.imwrite("roi_preview.jpg", frame)
            cv2.destroyWindow("Select ROI")

# Load first frame of video
video_path = r"sample.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Error loading video!")
    cap.release()
    exit()

frame = frame.copy()
cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", select_points)
cv2.waitKey(0)

# Save the ROI polygon points
with open("roi_polygon.json", "w") as f:
    json.dump(points, f)

print("ROI points saved to roi_polygon.json")
cap.release()
cv2.destroyAllWindows()
