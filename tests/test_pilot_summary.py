"""Fixed-threshold accuracy must not be inferred from EER or class imbalance."""
import pytest
from scripts.summarize_pilot import fixed_threshold_metrics


def test_perfect_ranking_can_have_bad_fixed_threshold_accuracy():
    result = fixed_threshold_metrics([0, 0, 1, 1], [.1, .2, .3, .4])
    assert result['eer_percent'] == 0
    assert result['auc'] == 1
    assert result['accuracy_percent'] == 50
    assert result['balanced_accuracy_percent'] == 50
    assert result['tp'] == 0 and result['fn'] == 2


def test_majority_accuracy_does_not_imply_balanced_detection():
    result = fixed_threshold_metrics([0]+[1]*9, [.9]*10)
    assert result['accuracy_percent'] == 90
    assert result['majority_baseline_accuracy_percent'] == 90
    assert result['balanced_accuracy_percent'] == 50
    assert result['fp'] == 1 and result['tn'] == 0
    with pytest.raises(ValueError):
        fixed_threshold_metrics([0, 1], [.2, 1.2])
