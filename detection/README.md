# Face Detection (YOLOv11)

Face detection module using a custom YOLOv11 implementation, trained on a single class: **face**.

## Overview

This folder contains:

- **Custom YOLOv11** – PyTorch implementation from scratch (`yolov11_custom.py`) with a drop-in interface compatible with Ultralytics-style training and inference (`yolo_interface.py`).
- **Training pipeline** – Dataset conversion (CSV → YOLO format), training, validation, and export (`main.py`).
- **Dataset config** – `data.yaml` defines train/val/test paths and the single class `face`.

## Structure

```
detection/
├── main.py           # Dataset conversion, training, validation, export
├── yolo_interface.py # YOLO wrapper (train/val/predict API)
├── yolov11_custom.py # YOLOv11 model implementation
├── data.yaml         # Dataset paths and class names
├── labels/           # YOLO-format labels (train, val, test)
├── images/           # Images (after running main.py with dataset)
├── dataset/          # Raw dataset (train/valid/test + CSV annotations)
├── yolov11_face/     # Training runs and weights (best.pt, last.pt)
└── README.md
```

## Dataset

By default this module is configured to use the **WIDER FACE YOLO dataset**:

- **Dataset**: [WIDER FACE for YOLO training](https://www.kaggle.com/datasets/lylmsc/wider-face-for-yolo-training)
- **Task**: single-class face detection (`face`)
- **Format**: YOLO text labels (`class x_center y_center width height`) with normalized coordinates

### Recommended layout

After downloading from Kaggle, extract the dataset under `detection/dataset/` so you end up with a standard YOLO layout, for example:

- `dataset/images/train/`
- `dataset/images/val/`
- `dataset/images/test/` (optional)
- `dataset/labels/train/`
- `dataset/labels/val/`
- `dataset/labels/test/` (optional)

Make sure `data.yaml` points to these folders and defines a single class:

- `nc: 1`
- `names: ["face"]`

### Using CSV-based datasets (optional)

You can still use CSV-based face-detection datasets (e.g. [Kaggle Face Detection](https://www.kaggle.com/datasets/adilshamim8/face-detection)) by arranging them as:

- `dataset/train/train/` – training images + `_annotations.csv`
- `dataset/valid/valid/` – validation images + `_annotations.csv`
- `dataset/test/test/` – test images + `_annotations.csv`

with CSV format: `filename, width, height, class, xmin, ymin, xmax, ymax`.  
`main.py` can be adapted to convert these CSVs to YOLO format (normalized `class x_center y_center width height`) and to write `data.yaml`.

## Requirements

- Python 3.x
- PyTorch
- OpenCV (`cv2`)
- Pandas, PyYAML, tqdm, matplotlib, Pillow

## Usage

1. **Prepare dataset**  
   Place your dataset in the `dataset/` folder with the structure above and `_annotations.csv` in each split.

2. **Convert and train**  
   Run the full pipeline (conversion, training, validation, sample inference, export):

   ```bash
   python main.py
   ```

   This will:

   - Convert CSV annotations to YOLO and create `images/` and `labels/`
   - Write/update `data.yaml`
   - Train YOLOv11 (e.g. `yolo11n.pt` by default) and save runs to `yolov11_face/`
   - Validate and run inference on sample images
   - Export the best model (e.g. ONNX) and copy `best.pt` / `last.pt` to `saved_model/`

3. **Inference**  
   Load the trained model and run prediction:

   ```python
   from yolo_interface import YOLO
   model = YOLO('yolov11_face/weights/best.pt')
   results = model('path/to/image.jpg')
   ```

## Training options (in `main.py`)

- **Model size**: `yolo11n.pt` (nano) by default; can use `yolo11s.pt`, `yolo11m.pt`, etc.
- **Epochs**, **batch size**, **image size**, **device** (GPU/CPU), and augmentation are set in the `model.train(...)` call in `main.py`.

## Outputs

- **Weights**: `yolov11_face/weights/best.pt`, `last.pt`
- **Plots**: training curves in the run folder
- **Exports**: ONNX (and optionally TensorRT/TFLite if uncommented)
- **Copied weights**: `saved_model/best.pt`, `saved_model/last.pt`
