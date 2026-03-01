# grade_from_key — MCQ Grading Module

Grade student answer sheets (output of `scoring.py`) against a teacher-supplied answer key file.

---

## Files

| File                  | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `grade_from_key.py`   | Main grading script                                     |
| `answer_key.json`     | Answer key — **fill in the correct answers before use** |
| `grading_report.json` | Output report (auto-generated after each run)           |

---

## Quick Start

From the **project root**:

```bash
python3 grade_from_key/grade_from_key.py <exam_class_id>
```

**Example:**

```bash
python3 grade_from_key/grade_from_key.py demo2
```

The script automatically:

- Reads scored sheets from `images/answer_sheets/demo2/ScoredSheets/`
- Loads the answer key from `grade_from_key/answer_key.json`
- Saves the report to `grade_from_key/grading_report.json`

---

## Step 1 — Fill in `answer_key.json`

Open `answer_key.json` and fill in the correct answers for each exam set code (`mã đề`):

```json
{
  "exam_name":       "Midterm Examination",
  "subject":         "Introduction to Computer Science",
  "total_questions": 60,
  "total_score":     10.0,
  "keys": {
    "423": ["ABC", "ACD", "ABCD", "x", "BC", "AD", ...],
    "915": ["A",   "B",   "C",   "x", "AB", "AC", ...]
  }
}
```

### Field reference

| Field             | Description                                                                           |
| ----------------- | ------------------------------------------------------------------------------------- |
| `exam_name`       | Name of the exam — printed in the console report header                               |
| `subject`         | Subject name — printed in the console report header                                   |
| `total_questions` | Number of questions per sheet (`20`, `40`, or `60`)                                   |
| `total_score`     | Maximum score (e.g. `10.0`). Each question is worth `total_score / total_questions`   |
| `keys`            | Map of exam set code → array of correct answers (length must equal `total_questions`) |

### Answer format

| Value    | Meaning                                              |
| -------- | ---------------------------------------------------- |
| `"A"`    | Single answer                                        |
| `"AB"`   | Multiple answers (letters in any order are accepted) |
| `"ABCD"` | All four options                                     |
| `"x"`    | Question intentionally left blank in the key         |

> **Note:** Letter order does not matter — `"BA"` and `"AB"` are treated as equal.

---

## Step 2 — Run `scoring.py` first

Make sure the answer sheets have been processed and scored:

```bash
python3 scoring.py <exam_class_id>
```

This creates `images/answer_sheets/<exam_class_id>/ScoredSheets/` with one `*_data.json` per sheet.

---

## Step 3 — Run the grading script

```bash
python3 grade_from_key/grade_from_key.py <exam_class_id>
```

---

## Output

### Console

```
╔════════════════════════════════════════════════════════════════════════╗
║                              GRADING REPORT                            ║
╚════════════════════════════════════════════════════════════════════════╝
  Exam    : Midterm Examination
  Subject : Introduction to Computer Science
  Scoring : exact match only  |  Total score = 10.0
────────────────────────────────────────────────────────────────────────

  Class: 247103  (9 student(s))
  Student Code     Exam Set      Score   Correct  Incorrect
  ······································································
  20193046         423            8.67        52          8
  ······································································
                     Average      8.67
                     Highest      8.67
                      Lowest      8.67

────────────────────────────────────────────────────────────────────────
  OVERALL  (10 students)
    Average score  : 7.87 / 10.0
    Highest        : 8.67
    Lowest         : 0.67
    Pass rate (≥50%): 9/10 (90.0%)
────────────────────────────────────────────────────────────────────────
```

### `grading_report.json`

Full machine-readable report containing per-student and per-question detail:

```json
{
  "exam_class_id": "demo2",
  "exam_name": "Midterm Examination",
  "total_score": 10.0,
  "results": [
    {
      "student_code": "20193046",
      "class_code":   "247103",
      "exam_code":    "423",
      "score":        8.67,
      "n_correct":    52,
      "n_incorrect":  8,
      "detail": [
        { "questionNo": 1, "student_ans": "ABC", "key_ans": "ABC", "earned": 0.1667, "verdict": "correct" },
        { "questionNo": 2, "student_ans": "ACD", "key_ans": "ABCD", "earned": 0.0, "verdict": "incorrect" },
        ...
      ]
    }
  ]
}
```

---

## Scoring rule

A question earns full mark **only if the student's answer exactly matches the key**. Any other answer — wrong, incomplete, extra letter, or blank — counts as **incorrect (0 points)**.

| Student answer | Key   | Result                 |
| -------------- | ----- | ---------------------- |
| `ABD`          | `ABD` | ✅ Correct — full mark |
| `AB`           | `ABD` | ❌ Incorrect — 0 pts   |
| `ABDC`         | `ABD` | ❌ Incorrect — 0 pts   |
| `` (blank)     | `ABD` | ❌ Incorrect — 0 pts   |

---

## Error messages

| Message                                     | Cause & fix                                               |
| ------------------------------------------- | --------------------------------------------------------- |
| `[ERROR] ScoredSheets folder not found`     | Run `scoring.py <exam_class_id>` first                    |
| `[ERROR] Answer key not found`              | Ensure `grade_from_key/answer_key.json` exists            |
| `⚠ No answer key found for exam code 'XYZ'` | Add exam set `"XYZ"` and its answers to `answer_key.json` |
