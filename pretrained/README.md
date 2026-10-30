# Trained compact-pilot checkpoints

These nine files are the actual best-dev checkpoints for three variants and seeds 0/1/2. Each checkpoint is under 0.5 MB and its SHA256 is recorded in `manifest.json`. They use **pilot_subset**, 3,000 training items, 1,000 dev items and at most eight epochs. Their cross-domain performance was poor; these are not full-benchmark models.

```python
from sdeseg.evaluate import load_model

model, training_state, device = load_model('pretrained/full_seed0.pt')
```

Scores use spoof=1 and bona fide=0. Inputs must follow the recorded four-second log-Mel preprocessing. The existing evaluator can load these files without retraining. The same hashes appear in the published final-evaluation logs. Checkpoints include the original training state for provenance; raw dataset audio is not included.

All three full-model seeds are published. Seed zero is a convenient example, not a test-selected winner. The full-data checkpoints from the interrupted earlier experiment are not substituted here.
