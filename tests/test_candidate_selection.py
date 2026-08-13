import numpy as np

from review.stage2.data import select_candidates


def test_candidate_threshold_and_label_free_fallback():
    assert select_candidates(np.array([0.2, 0.8, 0.76]), 0.75).tolist() == [1, 2]
    assert select_candidates(np.array([0.2, 0.6, 0.4]), 0.75).tolist() == [1]

