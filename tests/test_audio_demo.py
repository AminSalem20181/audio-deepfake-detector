"""Synthetic non-speech fixture checks demo plumbing, not model performance."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.demo_audio import demonstrate


def test_demo_creates_self_contained_report_without_training(tmp_path):
    path = tmp_path/'synthetic_tone.wav'
    sf.write(path, np.sin(np.arange(8000)*.04), 16000)
    result = demonstrate(path, Path('pretrained/full_seed0.pt'), tmp_path/'output', 'bonafide')
    assert 0 <= result['score_spoof'] <= 1 and result['threshold'] == .5
    assert result['scope'] == 'pilot_subset'
    assert result['prediction'] in ('spoof', 'bonafide')
    assert result['correct'] == (result['prediction'] == 'bonafide')
    assert json.loads((tmp_path/'output/prediction.json').read_text()) == result
    audio = sf.info(tmp_path/'output/processed.wav')
    assert audio.samplerate == 16000 and audio.frames == 64000
    page = (tmp_path/'output/report.html').read_text()
    assert 'data:image/png;base64,' in page and 'data:audio/wav;base64,' in page
    assert '<script' not in page
