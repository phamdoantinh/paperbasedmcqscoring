"""
grade_from_key.py — Grade scored MCQ answer sheets against an answer key file
=============================================================================
Reads:
  1. answer_key.json   — correct answers per exam set code (mã đề thi)
  2. ScoredSheets/     — JSON files output by scoring.py

Outputs:
  - Console: per-student score table grouped by class
  - grading_report.json: full machine-readable report

Usage
─────
  python3 grade_from_key/grade_from_key.py <exam_class_id>

  Example:
      python3 grade_from_key/grade_from_key.py demo2

  Reads  : images/answer_sheets/<exam_class_id>/ScoredSheets/  (project root)
  Key    : grade_from_key/answer_key.json
  Output : grade_from_key/grading_report.json

Answer key format  (answer_key.json)
──────────────────────────────────────────────────────────────────────
  {
    "exam_name":       "Midterm Exam",
    "subject":         "Introduction to Computer Science",
    "total_questions": 60,
    "total_score":     10.0,
    "keys": {
      "423": ["ABC", "ACD", "ABCD", ...],   // 60 answers for exam set 423
      "915": ["A",   "B",   "C",    ...]    // 60 answers for exam set 915
    }
  }

Scoring rule
────────────
  Exact match → full mark per question.
  Any other answer (wrong, incomplete, or unanswered) → 0 points.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Resolve project root (parent of this script's folder: grade_from_key/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = (s or "").strip().upper()
    return "".join(sorted(set(s)))   # deduplicate + sort: "ACBD" → "ABCD"


def _score_question(student_ans: str, key_ans: str,
                    mark: float) -> tuple[float, str]:
    """
    Returns (points_earned, verdict).
    verdict: 'correct' | 'incorrect'
    Rule: exact match only → full mark. Everything else (wrong, blank) → 0.
    """
    s = _norm(student_ans)
    k = _norm(key_ans)

    if s == k:           # covers both-blank (X==X) and normal exact match
        return mark, "correct"

    return 0.0, "incorrect"


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_answer_key(key_path: str) -> dict:
    with open(key_path, encoding="utf-8") as f:
        return json.load(f)


def load_scored_sheets(scored_dir: str) -> list[dict]:
    sheets = []
    for fname in sorted(os.listdir(scored_dir)):
        if not fname.endswith("_data.json"):
            continue
        path = os.path.join(scored_dir, fname)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # Attach source filename for traceability
        d["_source_file"] = fname
        sheets.append(d)
    return sheets


# ─────────────────────────────────────────────────────────────────────────────
# Grade one sheet
# ─────────────────────────────────────────────────────────────────────────────

def grade_sheet(sheet: dict, key_cfg: dict) -> dict:
    """
    Returns a grading result dict for a single student sheet.
    """
    exam_code    = sheet.get("testSetCode", "").strip()
    student_code = sheet.get("studentCode", "").strip()
    class_code   = sheet.get("examClassCode", "").strip()

    total_q   = key_cfg["total_questions"]
    total_pts = key_cfg["total_score"]
    mark_per_q = total_pts / total_q

    # Look up key for this exam set
    keys_map = key_cfg.get("keys", {})
    if exam_code not in keys_map:
        return {
            "student_code":  student_code,
            "class_code":    class_code,
            "exam_code":     exam_code,
            "error":         f"No answer key found for exam code '{exam_code}'",
            "score":         None,
            "source_file":   sheet.get("_source_file", ""),
        }

    correct_answers = keys_map[exam_code]   # list of strings, length = total_q

    # Build answer lookup from sheet
    student_answers = {a["questionNo"]: a["selectedAnswers"]
                       for a in sheet.get("answers", [])}

    details = []
    total_earned = 0.0
    n_correct    = 0
    n_incorrect  = 0

    for q_no in range(1, total_q + 1):
        key_ans     = correct_answers[q_no - 1] if q_no - 1 < len(correct_answers) else ""
        student_ans = student_answers.get(q_no, "")

        earned, verdict = _score_question(student_ans, key_ans, mark_per_q)
        total_earned += earned

        if verdict == "correct":
            n_correct   += 1
        else:
            n_incorrect += 1

        details.append({
            "questionNo":  q_no,
            "student_ans": _norm(student_ans) if student_ans else "",
            "key_ans":     _norm(key_ans),
            "earned":      round(earned, 4),
            "verdict":     verdict,
        })

    score = round(min(total_earned, total_pts), 2)

    return {
        "student_code": student_code,
        "class_code":   class_code,
        "exam_code":    exam_code,
        "score":        score,
        "total_score":  total_pts,
        "n_correct":    n_correct,
        "n_incorrect":  n_incorrect,
        "source_file":  sheet.get("_source_file", ""),
        "detail":       details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], key_cfg: dict):
    exam_name = key_cfg.get("exam_name", "")
    subject   = key_cfg.get("subject", "")
    total_pts = key_cfg.get("total_score", 10)
    W   = 72
    SEP = "─" * W

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  GRADING REPORT".center(W) + "║")
    print("╚" + "═" * W + "╝")
    if exam_name:
        print(f"  Exam    : {exam_name}")
    if subject:
        print(f"  Subject : {subject}")
    print(f"  Scoring : exact match only  |  Total score = {total_pts}")
    print(SEP)

    # Group by class
    by_class = defaultdict(list)
    for r in results:
        by_class[r["class_code"]].append(r)

    for class_code, students in sorted(by_class.items()):
        print(f"\n  Class: {class_code}  ({len(students)} student(s))")
        print(f"  {'Student Code':<16} {'Exam Set':<10} {'Score':>8}  "
              f"{'Correct':>8} {'Incorrect':>10}")
        print("  " + "·" * (W - 2))

        scores = []
        for r in sorted(students, key=lambda x: x["student_code"]):
            if r.get("error"):
                print(f"  {r['student_code']:<16} {r['exam_code']:<10}  "
                      f"⚠  {r['error']}")
                continue
            print(f"  {r['student_code']:<16} {r['exam_code']:<10} "
                  f"{r['score']:>7.2f}  "
                  f"{r['n_correct']:>8} {r['n_incorrect']:>10}")
            scores.append(r["score"])

        if scores:
            print("  " + "·" * (W - 2))
            avg = sum(scores) / len(scores)
            print(f"  {'Average':>26}  {avg:>7.2f}")
            print(f"  {'Highest':>26}  {max(scores):>7.2f}")
            print(f"  {'Lowest':>26}  {min(scores):>7.2f}")

    print()
    print(SEP)

    # Overall summary
    all_scores = [r["score"] for r in results if r.get("score") is not None]
    if all_scores:
        print(f"  OVERALL  ({len(all_scores)} students)")
        print(f"    Average score  : {sum(all_scores)/len(all_scores):.2f} / {total_pts}")
        print(f"    Highest        : {max(all_scores):.2f}")
        print(f"    Lowest         : {min(all_scores):.2f}")
        pass_count = sum(1 for s in all_scores if s >= total_pts * 0.5)
        print(f"    Pass rate (≥50%): {pass_count}/{len(all_scores)} "
              f"({pass_count/len(all_scores)*100:.1f}%)")
    print(SEP)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grade MCQ scored sheets against answer_key.json.")
    parser.add_argument("exam_class_id",
                        help="Exam class folder name (e.g. demo2). "
                             "Scored sheets are read from "
                             "images/answer_sheets/<exam_class_id>/ScoredSheets/")
    args = parser.parse_args()

    scored_dir = PROJECT_ROOT / "images" / "answer_sheets" / args.exam_class_id / "ScoredSheets"
    key_path   = Path(__file__).resolve().parent / "answer_key.json"
    out_path   = Path(__file__).resolve().parent / "grading_report.json"

    # Validate
    if not scored_dir.is_dir():
        print(f"[ERROR] ScoredSheets folder not found: {scored_dir}")
        sys.exit(1)
    if not key_path.is_file():
        print(f"[ERROR] Answer key not found: {key_path}")
        sys.exit(1)

    # Load
    key_cfg = load_answer_key(key_path)
    sheets  = load_scored_sheets(scored_dir)
    print(f"[INFO] Exam class   : {args.exam_class_id}")
    print(f"[INFO] Answer key   : {key_path}  "
          f"(exam sets: {list(key_cfg.get('keys', {}).keys())})")
    print(f"[INFO] Scored sheets: {len(sheets)} file(s) from {scored_dir}")

    # Grade
    results = [grade_sheet(s, key_cfg) for s in sheets]

    # Print
    print_report(results, key_cfg)

    # Save full report
    report = {
        "exam_class_id": args.exam_class_id,
        "exam_name":     key_cfg.get("exam_name", ""),
        "subject":       key_cfg.get("subject", ""),
        "scoring_rule":  "exact_match_only",
        "total_score":   key_cfg.get("total_score", 10),
        "results":       results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Full report saved → {out_path}")


if __name__ == "__main__":
    main()
