# Polygon-Based People Classification from Video using YOLOv8

This project allows you to classify people in a video as **employees** or **customers** based on whether they appear inside a user-defined polygonal region.

---

## 🚀 Features

- Polygon selection UI for defining ROI.
- YOLOv8 object detection.
- Real-time people classification.
- Counts employees and customers.
- Saves output as annotated video.

---

## 📂 Project Structure

```
.
├── select_roi.py              # Script to draw and save polygon ROI
├── polygon_person_classifier.py  # Script to detect and classify persons
├── roi_polygon.json           # JSON file containing selected polygon points
├── roi_preview.jpg            # Saved image with drawn polygon (optional)
└── output_video.avi           # Annotated output video
```

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- Ultralytics (`pip install ultralytics`)
- YOLOv8 model weights (`yolov8n.pt`)

---

## 🖱️ Usage

### 1. Select ROI Polygon
```bash
python select_roi.py
```
- Left-click to add points
- Right-click to finish selection
- This saves `roi_polygon.json` for later use

### 2. Run Detection & Classification
```bash
python polygon_person_classifier.py
```

---

## 📈 Business Use Case

By separating employees and customers in each frame and recording counts over time, you can answer:

- What hours have high footfall but low employee presence?
- Does customer-to-employee ratio affect sales?
- Are employees mostly inside their designated region?
- Are peak hours properly staffed?

You can extend this by logging frame timestamps and counts to CSV/DB for further analytics.

---

## 🔄 Adaptation

To use on another video:
1. Run `select_roi.py` on the new video to define a new polygon.
2. Use the new `roi_polygon.json` with `polygon_person_classifier.py`.

---

## 🧠 Future Enhancements

- Export results to CSV/Excel
- Daily/weekly heatmaps
- Pose detection for activity classification
- Integration with sales data

---

## 📸 Example Output

Annotated frame with green box = employee, blue = customer, polygon = employee zone.
