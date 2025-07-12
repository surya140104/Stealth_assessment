# ⚽ Player Re-Identification in Sports Footage (Liat.al Internship Task)

This project implements a real-time **player tracking and re-identification system** using a fine-tuned YOLOv11 object detection model and DeepSORT multi-object tracking.

> 🎯 **Objective**: Track each football player in a 15-second video, ensuring that players retain the same identity (ID) even after leaving and re-entering the frame.

---

## 📁 Repository Structure

```
├── stealth-intern-assessment.ipynb    # Main notebook (includes detection + tracking)
├── best.pt                            # Provided YOLOv11 detection model
├── ckpt.t7                            # DeepSORT ReID model
├── 15sec_input_720p.mp4               # Test video input
└── output_tracked_final.mp4           # Final result with IDs
```

---

## 🛠️ Setup & Dependencies

This project runs on **Kaggle** notebooks with the following dependencies:

- Python 3.10+
- OpenCV
- Torch & torchvision
- Ultralytics (YOLOv11)
- DeepSORT (custom repo)

```bash
pip install ultralytics opencv-python torch
```

---

## ▶️ How to Run

1. Upload the provided files:
   - `best.pt` to `/input/bestpt/`
   - `ckpt.t7` to `/input/ckptt7model/`
   - `15sec_input_720p.mp4` to `/input/videoframe/`
   - DeepSORT code to `/input/deepsortnnn/`

2. Run `stealth-intern-assessment.ipynb` from top to bottom.

3. The output video with tracked players (`Player {id}`) will be saved as:
   ```
   output_tracked_final.mp4
   ```

---

## 🧠 Approach

- **YOLOv11 (Ultralytics)** was used to detect players (`cls == 2`) per frame.
- Detected bounding boxes were passed to **DeepSORT**, which handles assigning unique IDs and tracking players even when they leave and return to the frame.
- Only player class was filtered for tracking to avoid ID pollution from referees or balls.

---

## ⚠️ Issue with Detection Model (`best.pt`)

During experimentation, it was consistently observed that the provided `best.pt` model:
- Detects only **1–2 players per frame**, even when many are clearly visible.
- Sometimes misclassifies other entities (e.g., referees) as players.
- This limits DeepSORT's ability to re-identify and consistently track all players.

📌 **Conclusion**: The core logic and tracker work correctly, but performance is constrained by the detection model. With a better-trained YOLOv11 model, tracking consistency and coverage would significantly improve.

---

## 🚀 Future Improvements

- Retrain or fine-tune the detection model on more diverse football frames.
- Add filtering or heuristics to suppress false positives.
- Improve occlusion handling or switch to more robust appearance models.

---

## 📧 Submission

Assignment submitted as part of the **AI Internship** with **Liat.al**  
Submit to: `arshdeep@liat.al` and `rishit@liat.al`

---
