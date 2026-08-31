"""Standalone evaluator support package (NO era dependency).

Modules:
  gpu           — GPU watchdog teardown + free-card selection.
  verdicts      — PASS / NOT PASS verdicts + REPORT.md from scores.jsonl.
  review_model  — optional review-web-app model builder (detection.json +
                  human/review_model.json with verdict-stamped displays).

The scoring closure itself lives under ``scorer/`` (the byte-frozen iter_038
recipe + its harness era_eval_common.py + serve_judge.py).
"""
