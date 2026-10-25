# Publishing on GitHub

Recommended repository name: `sde-seg-audio-deepfake-detector`.

From the project directory, create an empty GitHub repository with that name, then run:

```bash
git init -b main
git add .
git commit -m "Initial SDE-Seg implementation"
git remote add origin https://github.com/<YOUR_USERNAME>/sde-seg-audio-deepfake-detector.git
git push -u origin main
```

The `.gitignore` intentionally excludes raw datasets, checkpoints, feature caches, generated figures/logs, Python caches, and zip archives. Do not commit licensed audio or credentials.

Before publishing, run:

```bash
PYTHONPATH=. python -m pytest -q
git status --short
```

The current `REPORT.md` is intentionally honest about dataset-dependent metrics that have not been generated in this environment.
