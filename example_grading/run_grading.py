"""
run_grading.py — Self-contained grading demo
=============================================
Grades the pre-computed scored sheets in `scored_sheets/` against the
included `answer_key.json` and prints a summary report.

Run from anywhere — no arguments required:

    python3 example_grading/run_grading.py

Output
------
  - Console : per-student score table grouped by class
  - grading_output.json : full machine-readable grading report (written next
                          to this script, in example_grading/)
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# ── Paths (all relative to this script — works from any working directory) ──
HERE          = Path(__file__).resolve().parent
SCORED_DIR    = HERE / "scored_sheets"
KEY_PATH      = HERE / "answer_key.json"
OUTPUT_PATH   = HERE / "grading_output.json"
EXPECTED_PATH = HERE / "expected_output.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Normalise an answer string: strip, upper-case, deduplicate, sort letters."""
    return "".join(sorted(set((s or "").strip().upper())))


def score_question(student_ans: str, key_ans: str, mark: float):
    s, k = _norm(student_ans), _norm(key_ans)
    return (mark, "correct") if s == k else (0.0, "incorrect")


def load_sheets(folder: Path) -> list:
    sheets = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith("_data.json"):
            continue
        with open(folder / fname, encoding="utf-8") as f:
            d = json.load(f)
        d["_source"] = fname
        sheets.append(d)
    return sheets


def grade(sheet: dict, key_cfg: dict) -> dict:
    exam_code    = sheet.get("testSetCode", "").strip()
    student_code = sheet.get("studentCode", "").strip()
    class_code   = sheet.get("examClassCode", "").strip()

    total_q   = key_cfg["total_questions"]
    total_pts = key_cfg["total_score"]
    mark_q    = total_pts / total_q

    keys_map = key_cfg.get("keys", {})
    if class_code not in keys_map or exam_code not in keys_map[class_code]:
        return {
            "student_code": student_code, "class_code": class_code,
            "exam_code": exam_code,
            "error": f"No answer key for exam code '{exam_code}' in class '{class_code}'",
            "score": None, "source_file": sheet["_source"],
        }

    correct_answers = keys_map[class_code][exam_code]
    student_answers = {a["questionNo"]: a["selectedAnswers"]
                       for a in sheet.get("answers", [])}

    details, total_earned, n_correct, n_incorrect = [], 0.0, 0, 0
    for q in range(1, total_q + 1):
        key_ans = correct_answers[q - 1] if q - 1 < len(correct_answers) else ""
        stu_ans = student_answers.get(q, "")
        earned, verdict = score_question(stu_ans, key_ans, mark_q)
        total_earned += earned
        (n_correct if verdict == "correct" else n_incorrect).__class__  # noqa
        if verdict == "correct":
            n_correct += 1
        else:
            n_incorrect += 1
        details.append({
            "questionNo": q,
            "student_ans": _norm(stu_ans),
            "key_ans": _norm(key_ans),
            "earned": round(earned, 4),
            "verdict": verdict,
        })

    return {
        "student_code": student_code, "class_code": class_code,
        "exam_code": exam_code,
        "score": round(min(total_earned, total_pts), 2),
        "total_score": total_pts,
        "n_correct": n_correct, "n_incorrect": n_incorrect,
        "source_file": sheet["_source"],
        "detail": details,
    }


def print_report(results: list, key_cfg: dict):
    W = 72
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  GRADING REPORT".center(W) + "║")
    print("╚" + "═" * W + "╝")
    print(f"  Exam    : {key_cfg.get('exam_name', '')}")
    print(f"  Subject : {key_cfg.get('subject', '')}")
    print(f"  Scoring : exact match only  |  Total score = {key_cfg['total_score']}")
    print("─" * W)

    by_class = defaultdict(list)
    for r in results:
        by_class[r["class_code"]].append(r)

    for cls, students in sorted(by_class.items()):
        print(f"\n  Class: {cls}  ({len(students)} student(s))")
        print(f"  {'Student Code':<16} {'Exam Set':<10} {'Score':>8}  "
              f"{'Correct':>8} {'Incorrect':>10}")
        print("  " + "·" * (W - 2))
        scores = []
        for r in sorted(students, key=lambda x: x["student_code"]):
            if r.get("error"):
                print(f"  {r['student_code']:<16} {r['exam_code']:<10}  ⚠  {r['error']}")
                continue
            print(f"  {r['student_code']:<16} {r['exam_code']:<10} "
                  f"{r['score']:>7.2f}  {r['n_correct']:>8} {r['n_incorrect']:>10}")
            scores.append(r["score"])
        if scores:
            print("  " + "·" * (W - 2))
            avg = sum(scores) / len(scores)
            print(f"  {'Average':>26}  {avg:>7.2f}")
            print(f"  {'Highest':>26}  {max(scores):>7.2f}")
            print(f"  {'Lowest':>26}  {min(scores):>7.2f}")

    print()
    print("─" * W)
    all_scores = [r["score"] for r in results if r.get("score") is not None]
    if all_scores:
        total_pts = key_cfg["total_score"]
        pass_count = sum(1 for s in all_scores if s >= total_pts * 0.5)
        print(f"  OVERALL  ({len(all_scores)} students)")
        print(f"    Average score  : {sum(all_scores)/len(all_scores):.2f} / {total_pts}")
        print(f"    Highest        : {max(all_scores):.2f}")
        print(f"    Lowest         : {min(all_scores):.2f}")
        print(f"    Pass rate (≥50%): {pass_count}/{len(all_scores)} "
              f"({pass_count/len(all_scores)*100:.1f}%)")
    print("─" * W)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Scored sheets : {SCORED_DIR}")
    print(f"[INFO] Answer key    : {KEY_PATH}")

    with open(KEY_PATH, encoding="utf-8") as f:
        key_cfg = json.load(f)

    sheets  = load_sheets(SCORED_DIR)
    print(f"[INFO] Loaded {len(sheets)} scored sheet(s) | "
          f"Exam classes in key: {list(key_cfg['keys'].keys())}")

    results = [grade(s, key_cfg) for s in sheets]
    print_report(results, key_cfg)

    report = {
        "exam_name":    key_cfg.get("exam_name", ""),
        "subject":      key_cfg.get("subject", ""),
        "total_score":  key_cfg["total_score"],
        "results":      results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Full report saved → {OUTPUT_PATH}")

    # Compare with expected output if present
    if EXPECTED_PATH.exists():
        with open(EXPECTED_PATH, encoding="utf-8") as f:
            expected = json.load(f)
        exp_scores = {r["student_code"]: r["score"]
                      for r in expected["results"] if r.get("score") is not None}
        got_scores = {r["student_code"]: r["score"]
                      for r in results if r.get("score") is not None}
        mismatches = [(k, exp_scores.get(k), got_scores.get(k))
                      for k in set(exp_scores) | set(got_scores)
                      if exp_scores.get(k) != got_scores.get(k)]
        if mismatches:
            print("[WARN] Score mismatches vs expected_output.json:")
            for student, exp, got in mismatches:
                print(f"       Student {student}: expected {exp}, got {got}")
        else:
            print("[OK]  All scores match expected_output.json ✓")


if __name__ == "__main__":
    main()
