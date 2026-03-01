# Paper-Based MCQ Scoring System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://docs.ultralytics.com/models/yolov8/)

> **This is the `yolov8` branch — the original implementation as described in the published paper.**
> A newer version using YOLOv11 and three separate specialized models is available on the `main` branch.

An automated optical scoring system for paper-based multiple-choice question (MCQ) answer sheets. The system uses computer vision and deep learning (YOLOv8) to detect alignment markers, extract student/exam information, and recognize selected answers from scanned or photographed answer sheet images — producing structured JSON output suitable for downstream grading pipelines.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Answer Sheet Template](#answer-sheet-template)
- [Usage](#usage)
  - [Preparing Input Images](#preparing-input-images)
  - [Running the Scoring Pipeline](#running-the-scoring-pipeline)
  - [Output Description](#output-description)
- [Models](#models)
- [Grading With Answer Key](#grading-with-answer-key)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This system automates the grading of paper-based MCQ exams. Given a folder of answer sheet images (JPEG or PNG), it:

1. **Detects alignment markers** on the answer sheet to correct skew and perspective.
2. **Extracts student information** (class code, student code, exam/test-set code) from the information zone.
3. **Recognizes selected answers** for each question (supporting up to 60 questions per sheet with multi-answer combinations A, B, C, D, AB, AC, …, ABCD).
4. **Writes annotated output images** and structured **JSON result files** per answer sheet.
5. **Logs potentially uncertain predictions** (low-confidence detections) to a warning file.

A single YOLOv8 model (`best.pt`) trained on all 29 classes handles all three detection tasks: markers, student info digits, and answer bubbles. The pipeline is designed for integration with an e-learning support platform but can also be used as a standalone batch-processing tool.

---

## Features

- ✅ Automatic skew correction using marker-based perspective transform
- ✅ Single unified YOLOv8 model covers all detection tasks
- ✅ Supports 20, 40, and 60 question answer sheets
- ✅ Multi-answer recognition (single and combination choices: AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD)
- ✅ Student information zone OCR (class code, student ID, test-set code)
- ✅ Confidence-based warning system for low-certainty predictions
- ✅ JSON output format for easy downstream integration
- ✅ Annotated output images highlighting detected answers

---

## System Architecture

```
Input image (JPG/PNG)
        │
        ▼
┌──────────────────────────┐
│   Marker Detection       │  ← best.pt  (classes: marker1, marker2)
│   & Image Alignment      │    Detect 4 markers → calculate rotation angle
│   (get_marker)           │    → warpAffine → warpPerspective → crop doc
└───────────┬──────────────┘
            │  Corrected & cropped document  (resized to 1056 × 1500 px)
            ▼
┌──────────────────────────┐      ┌────────────────────────────┐
│  Info Zone Crop          │ ───► │  Info Recognition          │  ← best.pt
│  x: 500–1006, y: 0–500   │      │  (predictInfo)             │  (classes: 0–9, x)
│  → resize to 640 × 640   │      │  → class_code, student_code│
└──────────────────────────┘      │     exam_code              │
                                  └─────────────┬──────────────┘
                                                │
┌──────────────────────────┐      ┌─────────────▼──────────────┐
│  Answer Column Crops     │ ───► │  Answer Recognition        │  ← best.pt
│  3 columns at x=30,350,  │      │  (predictAnswer)           │  (classes: unchoice,
│  660; y=480; 350×896 px  │      │  → per-question answer     │        A–ABCD)
│  → resize to 250 × 640   │      │     array                  │
└──────────────────────────┘      └─────────────┬──────────────┘
                                                │
                               ┌────────────────▼───────────────┐
                               │  JSON Output +                 │
                               │  Annotated Image (mergeImages) │
                               └────────────────────────────────┘
```

![System Flow](docs/StructureDiagram.png)

**Key modules:**

| File                               | Description                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `scoring.py`                       | Main pipeline: marker detection, image alignment, info/answer prediction, output writing        |
| `utils.py`                         | All utilities: geometry, perspective transform, angle calculation, class mapping, image helpers |
| `grade_from_key/grade_from_key.py` | Standalone grading script: compare scored sheets against an answer key file                     |

---

## Requirements

- Python **3.8** or higher
- The following Python packages (see also `requirements.txt`):

| Package                  | Version  | Purpose                |
| ------------------------ | -------- | ---------------------- |
| `opencv-python-headless` | 4.9.0.80 | Image processing       |
| `ultralytics`            | ≥ 8.0    | YOLOv8 model inference |
| `numpy`                  | ≥ 1.21   | Numerical operations   |

> **Note:** `Flask` and `uwsgi` are commented out in `requirements.txt`. They are only needed if you plan to deploy the system as a REST API web service.

---

## Installation

### 1. Clone the repository

```bash
git clone -b yolov8 https://github.com/<your-username>/paper-based-mcq-scoring.git
cd paper-based-mcq-scoring
```

### 2. (Optional) Create a virtual environment

Using a virtual environment is not required, but it is recommended to avoid conflicts with other packages already installed on your machine.

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

> You can skip this step and install dependencies directly into your system Python if preferred.

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install ultralytics numpy
```

### 4. Verify the model file

Ensure the single YOLOv8 model weight file is present in the `Model/` directory:

```
Model/
└── best.pt      # Unified YOLOv8 detector (all 29 classes)
```

> The model file is **not** included in this repository due to its size. Please contact the authors or download it from the provided release assets.

---

## Directory Structure

```
paper-based-mcq-scoring/
│
├── Model/
│   └── best.pt                         # Pre-trained YOLOv8 weights (all tasks)
│
├── images/
│   └── answer_sheets/
│       └── <exam_class_id>/            # One folder per exam session
│           ├── 1.jpg                   # Input answer sheet images
│           ├── 2.jpg
│           ├── ...
│           ├── HandledSheets/          # (auto-created) Annotated output images
│           ├── ScoredSheets/           # (auto-created) JSON result files
│           └── MayBeWrong/             # (auto-created) Low-confidence warning log
│
├── scoring.py                          # Main scoring pipeline
├── utils.py                            # All utility functions (geometry, labels, image helpers)
├── grade_from_key/                     # Grading module
│   ├── grade_from_key.py               # Script: compare scored sheets against answer key
│   ├── answer_key.json                 # Answer key (fill in correct answers per exam set)
│   └── grading_report.json             # (auto-generated) Grading output report
├── docs/                               # Documentation assets
│   ├── AnswerSheetTemplate.pdf         # Printable answer sheet template
│   ├── AnswerSheetTemplate.png         # Answer sheet template image
│   └── StructureDiagram.png            # System architecture diagram
├── requirements.txt
└── README.md
```

---

## Answer Sheet Template

The file `docs/AnswerSheetTemplate.pdf` is the official printable template that this system is designed to process. Print it on **A4 paper** before scanning or photographing.

### Layout Overview

![Answer Sheet Template](docs/AnswerSheetTemplate.png)

### Printing Notes

- Print at **100% scale** on **A4 (210 × 297 mm)** — do **not** scale to fit
- Use a **laser printer** for best marker contrast
- Ensure all 4 alignment markers are fully printed and not clipped by the page margin

---

## Usage

### Preparing Input Images

1. Create a folder named after the **exam class ID** inside `images/answer_sheets/`:

```bash
mkdir -p images/answer_sheets/<exam_class_id>
```

2. Place all scanned or photographed answer sheet images (`.jpg`, `.jpeg`, or `.png`) inside that folder.

**Image requirements:**

- The answer sheet must contain **4 alignment markers**: 3 × `marker1` (at top-left, top-right, bottom-left) and 1 × `marker2` (at bottom-right).
- Recommended image resolution: **≥ 1056 × 1500 px**.
- Supported formats: `JPEG`, `PNG`.

---

### Running the Scoring Pipeline

Run the main script from the project root, passing the exam class folder name as the argument:

```bash
python3 scoring.py <exam_class_id>
```

**Example:**

```bash
python3 scoring.py demo2
```

This will process all images inside `images/answer_sheets/demo2/` and write results to the automatically created subdirectories.

---

### Output Description

For each successfully processed answer sheet image (e.g., `1.jpg`), the system produces:

#### 1. JSON Result File — `ScoredSheets/<filename>_data.json`

```json
{
  "examClassCode": "demo2",
  "studentCode": "026983557",
  "testSetCode": "014",
  "answers": [
    { "questionNo": 1, "selectedAnswers": "A" },
    { "questionNo": 2, "selectedAnswers": "BC" },
    { "questionNo": 3, "selectedAnswers": "x" },
    ...
    { "questionNo": 60, "selectedAnswers": "D" }
  ],
  "handledScoredImg": "images/answer_sheets/demo2/HandledSheets/handled_1.jpg",
  "originalImg": "images/answer_sheets/demo2/1.jpg",
  "originalImgFileName": "1.jpg"
}
```

| Field                       | Type      | Description                                                                                                |
| --------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| `examClassCode`             | `string`  | Detected class/course code from the info zone                                                              |
| `studentCode`               | `string`  | Detected student ID number                                                                                 |
| `testSetCode`               | `string`  | Detected test/exam set code (3 digits)                                                                     |
| `answers`                   | `array`   | List of per-question answer objects                                                                        |
| `answers[].questionNo`      | `integer` | Question number (1-indexed)                                                                                |
| `answers[].selectedAnswers` | `string`  | Selected answer(s): `"A"`, `"B"`, `"C"`, `"D"`, combinations like `"AB"`, `"BCD"`, or `"x"` for unanswered |
| `handledScoredImg`          | `string`  | Path to the annotated output image                                                                         |
| `originalImg`               | `string`  | Path to the original input image                                                                           |
| `originalImgFileName`       | `string`  | File name of the original input image                                                                      |

#### 2. Annotated Image — `HandledSheets/handled_<filename>.<ext>`

A copy of the answer sheet with colored bounding boxes drawn over detected answers:

- 🟢 **Green box**: high-confidence prediction (conf ≥ 0.79)
- 🟠 **Orange box**: low-confidence prediction (also logged to warning file)

#### 3. Warning Log — `MayBeWrong/maybe_wrong.txt`

If any detection has a confidence score below the threshold (`0.79` by default), one line per warning is written:

```
[LOW CONF] Answer zone | File: t2.jpg | Question 5 | Predicted: "A" | Conf: 0.71
[LOW CONF] Info zone   | File: t2.jpg | Column 4 (left→right) | Predicted: "x" | Conf: 0.68
```

Each record is a single line with `|`-separated fields: zone type, filename, location, predicted label, and confidence score.

---

## Grading With Answer Key

After scoring, use the grading module to compare detected answers against the answer key and compute each student's score.

📄 **Full instructions → [`grade_from_key/README.md`](grade_from_key/README.md)**

**Quick start:**

```bash
# 1. Fill in the correct answers per exam set code
nano grade_from_key/answer_key.json

# 2. Run the grading script
python3 grade_from_key/grade_from_key.py <exam_class_id>
```

Output is printed to the console and saved to `grade_from_key/grading_report.json`.

---

## Models

This branch uses a **single unified YOLOv8 model** (`best.pt`) trained on all 29 classes across three detection tasks simultaneously:

| Class index | Class label | Task             |
| ----------- | ----------- | ---------------- |
| 0           | `0000`      | Answer bubble    |
| 1           | `1000`      | Answer bubble    |
| 2           | `0100`      | Answer bubble    |
| 3           | `0010`      | Answer bubble    |
| 4           | `0001`      | Answer bubble    |
| 5           | `1100`      | Answer bubble    |
| 6           | `1010`      | Answer bubble    |
| 7           | `1001`      | Answer bubble    |
| 8           | `0110`      | Answer bubble    |
| 9           | `0101`      | Answer bubble    |
| 10          | `0011`      | Answer bubble    |
| 11          | `1110`      | Answer bubble    |
| 12          | `1101`      | Answer bubble    |
| 13          | `1011`      | Answer bubble    |
| 14          | `0111`      | Answer bubble    |
| 15          | `1111`      | Answer bubble    |
| 16          | `0`         | Info digit       |
| 17          | `1`         | Info digit       |
| 18          | `2`         | Info digit       |
| 19          | `3`         | Info digit       |
| 20          | `4`         | Info digit       |
| 21          | `5`         | Info digit       |
| 22          | `6`         | Info digit       |
| 23          | `7`         | Info digit       |
| 24          | `8`         | Info digit       |
| 25          | `9`         | Info digit       |
| 26          | `unchoice`  | Info: blank cell |
| 27          | `marker1`   | Alignment marker |
| 28          | `marker2`   | Alignment marker |

The model is a custom-trained **YOLOv8** detector on a dataset of Vietnamese university MCQ answer sheets. The training methodology is described in the published paper (see [Citation](#citation)).

> For the newer implementation with three specialized YOLOv11 models, see the `main` branch.

---

## Configuration

Key parameters that can be adjusted directly in the source files:

| Parameter           | Location     | Default           | Description                                                           |
| ------------------- | ------------ | ----------------- | --------------------------------------------------------------------- |
| `threshold_warning` | `utils.py`   | `0.79`            | Confidence threshold below which a prediction is flagged as uncertain |
| `numberAnswer`      | `scoring.py` | `60`              | Number of questions per answer sheet (supported: `20`, `40`, `60`)    |
| `pWeight`           | `scoring.py` | `./Model/best.pt` | Path to the unified YOLOv8 model                                      |

**Image crop coordinates** (fixed for the standard answer sheet layout, defined in `utils.py`):

| Region          | x range  | y range  | Resized to |
| --------------- | -------- | -------- | ---------- |
| Info zone       | 500–1006 | 0–500    | 640 × 640  |
| Answer column 1 | 30–380   | 480–1376 | 250 × 640  |
| Answer column 2 | 350–700  | 480–1376 | 250 × 640  |
| Answer column 3 | 660–1010 | 480–1376 | 250 × 640  |

---

## Dataset

The training and evaluation dataset for this system is publicly available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18816315.svg)](https://doi.org/10.5281/zenodo.18816315)

**Dataset:** [https://doi.org/10.5281/zenodo.18816315](https://doi.org/10.5281/zenodo.18816315)

The dataset contains labelled answer sheet images used to train and evaluate the YOLOv8 model for marker detection, student info recognition, and answer bubble classification.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Pham Doan Tinh, Ta Quang Minh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

This software is the implementation of the following peer-reviewed publication. If you use this system in academic work, please cite:

**Pham Doan Tinh and Ta Quang Minh**, "Automated Paper-based Multiple Choice Scoring Framework using Fast Object Detection Algorithm," _International Journal of Advanced Computer Science and Applications (IJACSA)_, vol. 15, no. 1, 2024. DOI: [10.14569/IJACSA.2024.01501115](http://dx.doi.org/10.14569/IJACSA.2024.01501115)

```bibtex
@article{Tinh2024,
  title     = {Automated Paper-based Multiple Choice Scoring Framework using Fast Object Detection Algorithm},
  journal   = {International Journal of Advanced Computer Science and Applications},
  doi       = {10.14569/IJACSA.2024.01501115},
  url       = {http://dx.doi.org/10.14569/IJACSA.2024.01501115},
  year      = {2024},
  publisher = {The Science and Information Organization},
  volume    = {15},
  number    = {1},
  author    = {Pham Doan Tinh and Ta Quang Minh}
}
```

---

## Contact

For questions, issues, or contributions, please open a GitHub Issue or contact the authors:

- **Pham Doan Tinh** — corresponding author
- **Ta Quang Minh**

Paper available at: [https://thesai.org/Publications/ViewPaper?Volume=15&Issue=1&Code=IJACSA&SerialNo=115](https://thesai.org/Publications/ViewPaper?Volume=15&Issue=1&Code=IJACSA&SerialNo=115)
