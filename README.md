# NGALANGIN: Navigasi & Analisis Luas Area Gangguan Infrastruktur

A real-time sidewalk accessibility auditing system powered by Computer Vision. NGALANGIN detects sidewalk obstructions (primarily _gerobak_ — mobile vendor carts) using instance segmentation and measures the remaining walkable area through homography-based Bird's Eye View (BEV) transformation.

## Features

- **Dual-Model Instance Segmentation** — YOLOv11-based obstacle detector (Model A) combined with a general environment segmenter (Model B) for robust sidewalk boundary extraction
- **FAST Feature Verification** — Uses FAST keypoint detection (LO2) to validate obstacle masks and reduce false positives
- **Homography & BEV Mapping** — Perspective transform maps detected masks onto a top-down view for accurate area measurement in real-world units (m²)
- **Real-Time Accessibility Scoring** — Computes sidewalk accessibility percentage by comparing clear vs. obstructed area
- **Interactive Streamlit Dashboard** — Upload images/videos or use a webcam for live analysis with configurable ROI, confidence thresholds, and visual overlays
- **Model Evaluation Tab** — Built-in training metrics, confusion matrices, PR curves, and F1 analysis

## Tech Stack

| Component        | Technology                                    |
| ---------------- | --------------------------------------------- |
| Language         | Python 3.10+                                  |
| Framework        | Streamlit                                     |
| CV Models        | YOLOv11 (Ultralytics)                         |
| Image Processing | OpenCV, NumPy                                 |
| Deep Learning    | PyTorch                                       |
| Dataset Source   | Custom dataset (Jalan Rawa Belong) & Roboflow |

## Setup & Installation

```bash
# Clone the repository
git clone <repository-url>
cd cv-code

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Place model weight files in the `models/` directory:

- `models/yolov11_seq_obstacle.pt` — Fine-tuned obstacle segmentation model
- `models/yolov11x_seq_environment.pt` — Pre-trained environment segmentation model

## Running the Application

```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

## Project Structure

```
cv-code/
├── streamlit_app.py              # Main Streamlit application entry point
├── config.yaml                   # Model and pipeline configuration
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Core source modules
│   ├── perception.py             #   Dual-model YOLO segmentation pipeline
│   ├── features.py               #   FAST keypoint detection & verification
│   ├── geometry.py               #   Homography, BEV warp, area computation
│   └── processing.py             #   Morphological mask cleaning
│
├── models/                       # Model weights & training artifacts (partially gitignored)
│   ├── yolov11_seq_obstacle.pt         # Fine-tuned obstacle model
│   └── yolov11x_seq_environment.pt     # Pre-trained environment model
│
├── data/                         # Datasets & raw video footage (partially gitignored)
│   ├── NGALANGIN_v3_dataset/     #   Extracted COCO dataset from Jalan Rawa Belong
│   ├── raw_videos/               #   Raw sidewalk video recordings
```

## Dataset

- **Source**: Custom dataset collected from several 30fps video recordings at Jalan Rawa Belong, then annotated and exported via Roboflow (COCO Segmentation format)
- **Classes**: `gerobak` (sidewalk vendor carts), `drivable_road`, `obstruction`
- **Preprocessing**: Annotation in COCO format, augmentation via Albumentations

## Methodology

1. **Image Preprocessing** — Standard operators: resizing, normalization (LO1)
2. **Feature Detection** — FAST corner detector for mask verification (LO2)
3. **Instance Segmentation** — Dual YOLOv11 models for obstacle and environment segmentation (LO3)
4. **Spatial Analysis** — Homography-based BEV transformation for real-world area measurement (LO4)
5. **Accessibility Scoring** — Percentage of unobstructed sidewalk area

## Team Members

- Anandhio Varistama
- Ganesha Chandra Abiwardhana
- Ivan Novanto Bastian
- Jason Tirta
- Muhammad Rizki Akbar

## References

- Ultralytics YOLOv11 Documentation
- OpenCV FAST Feature Detector
- Roboflow Dataset Platform

---

_BINUS University — Computer Vision (Semester 4) — AOL Final Project_
