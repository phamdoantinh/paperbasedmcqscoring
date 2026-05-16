# Reproducing the Reported Results

This guide walks you through reproducing the scoring and grading results reported in the paper, using the **sample answer sheets and answer key already included in this repository**.

> **Want to test just the grading module quickly (no model/images needed)?**
> See [`example_grading/README.md`](example_grading/README.md) — a single command is enough.

---

## What Is Included in This Repository

| Resource                 | Path                                         | Description                                              |
| ------------------------ | -------------------------------------------- | -------------------------------------------------------- |
| Pretrained model weights | `Model/best.pt`                              | Fine-tuned YOLOv8m detector (29 classes, ~52 MB)        |
| Sample answer sheets     | `images/demo1/`                              | 10 real scanned answer sheet images                      |
| Example answer key       | `grade_from_key/answer_key.json`             | Correct answers for exam sets 101, 102, 568, 423         |
| Expected scored output   | `images/demo1/ScoredSheets/`                 | Pre-computed JSON result per sheet                       |
| Expected grading report  | `grade_from_key/grading_report.json`         | Pre-computed grading report for `demo1`                  |
| Grading demo (standalone)| `example_grading/`                           | Self-contained grading example — no model/images needed  |

> **Note on the model:** `best.pt` was obtained by fine-tuning the publicly available **YOLOv8m** pretrained weights (`yolov8m.pt`) from [Ultralytics](https://github.com/ultralytics/ultralytics) on our custom dataset of Vietnamese university MCQ answer sheets. The training dataset is available on Zenodo: [https://doi.org/10.5281/zenodo.18816315](https://doi.org/10.5281/zenodo.18816315).

---

## Option A — Quick Grading Demo (No Model or Images Required)

To reproduce only the **grading step** without running the full image-processing pipeline:

```bash
git clone https://github.com/phamdoantinh/paperbasedmcqscoring.git
cd paperbasedmcqscoring
python3 example_grading/run_grading.py
```

This uses pre-scored JSON files already included in `example_grading/scored_sheets/` and verifies the output against `example_grading/expected_output.json`. No dependencies beyond the Python standard library are required.

📄 Full details → [`example_grading/README.md`](example_grading/README.md)

---

## Option B — Full Pipeline (Scoring + Grading)

### Step 1 — Clone and Install

```bash
git clone https://github.com/phamdoantinh/paperbasedmcqscoring.git
cd paperbasedmcqscoring
pip install -r requirements.txt
pip install ultralytics
```

### Step 2 — Run the Scoring Pipeline

Process the 10 sample answer sheets in `demo1`:

```bash
python3 scoring.py demo1
```

This reads images from `images/demo1/`, runs detection with `Model/best.pt`, and writes results to:
- `images/demo1/ScoredSheets/` — one JSON file per sheet
- `images/demo1/HandledSheets/` — annotated images with bounding boxes
- `images/demo1/MayBeWrong/maybe_wrong.txt` — low-confidence warnings

### Step 3 — Run the Grading Script

```bash
python3 grade_from_key/grade_from_key.py demo1
```

The answer key (`grade_from_key/answer_key.json`) already contains the correct answers for all exam sets present in `demo1` (sets `101`, `102`, `568`, `423`). No manual editing is needed.

The grading report is printed to the console and saved to `grade_from_key/grading_report.json`.

### Step 4 — Verify Against Expected Output

Compare the generated report against the pre-computed reference included in the repository:

```bash
# The expected grading report is already in the repo:
# grade_from_key/grading_report.json

# Re-run and compare:
python3 grade_from_key/grade_from_key.py demo1
```

Expected summary for `demo1` (9 students, 4 exam sets):

```
  OVERALL  (9 students)
    Average score  : 8.50 / 10.0
    Highest        : 10.00
    Lowest         : 1.67
    Pass rate (≥50%): 8/9 (88.9%)
```

---

## Annotated Output Example

After running Step 2, annotated images are saved in `HandledSheets/`. Green boxes indicate high-confidence predictions (≥ 0.79); orange boxes indicate low-confidence predictions also logged to `MayBeWrong/maybe_wrong.txt`.

---

## Troubleshooting

| Error | Fix |
| -------------------------------------------- | --------------------------------------------------- |
| `ModuleNotFoundError: ultralytics`            | Run `pip install ultralytics`                       |
| `[ERROR] ScoredSheets folder not found`       | Run `scoring.py demo1` before `grade_from_key.py`  |
| `⚠ No answer key found for exam code 'XYZ'`  | Add exam set `XYZ` to `grade_from_key/answer_key.json` |
