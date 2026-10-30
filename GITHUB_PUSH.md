# Publishing on GitHub

The configured repository is `git@github.com:AminSalem20181/audio-deepfake-detector.git`, on branch `master`. For subsequent updates from this existing checkout:

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short
git add .
git commit -m "Describe the concrete change"
git push origin master
```

The `.gitignore` excludes raw datasets, caches, runtime/debug logs, large archives and unselected training checkpoints. It explicitly allows the nine small best-dev models in `pretrained/`, final pilot evidence, selected training histories and presentation figures. No raw dataset audio or credentials should be committed.

Before publishing, run:

```bash
PYTHONPATH=. python -m pytest -q
git status --short
```

`REPORT.md` records completed compact-pilot measurements and their limitations. `docs/PROFESSOR_BRIEF_FA.md` explains the work in Persian. To regenerate reporting metrics and plots from published predictions, run `python scripts/summarize_pilot.py`; it never retrains or scores audio. A summary's Accuracy at threshold 0.5 is distinct from EER.
