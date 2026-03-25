"""
tests/test_tracker.py — Unit tests for PlateTracker.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from utils.tracker import PlateTracker, iou


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestIou:
    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou((0, 0, 50, 50), (60, 60, 110, 110)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < score < 1.0

    def test_contained_box(self):
        score = iou((10, 10, 90, 90), (0, 0, 100, 100))
        assert score > 0.6


# ---------------------------------------------------------------------------
# PlateTracker
# ---------------------------------------------------------------------------

DET = lambda plate: (100, 100, 300, 150, 0.9, plate)   # helper


class TestPlateTracker:
    def test_no_confirmation_before_threshold(self):
        tracker = PlateTracker(confirm_frames=3)
        events = tracker.update([DET("KL07BB1234")])
        assert not any(e["type"] == "confirmed" for e in events)

    def test_confirms_after_threshold(self):
        tracker = PlateTracker(confirm_frames=3)
        events = []
        for _ in range(3):
            events += tracker.update([DET("KL07BB1234")])
        confirmed = [e for e in events if e["type"] == "confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0]["plate"] == "KL07BB1234"

    def test_only_emits_once(self):
        tracker = PlateTracker(confirm_frames=2)
        events = []
        for _ in range(5):
            events += tracker.update([DET("MH12AB3456")])
        confirmed = [e for e in events if e["type"] == "confirmed"]
        assert len(confirmed) == 1

    def test_lost_event_emitted(self):
        tracker = PlateTracker(confirm_frames=2, max_lost=2)
        for _ in range(2):
            tracker.update([DET("KL07BB1234")])   # confirm it
        events = []
        for _ in range(3):   # stop sending detections
            events += tracker.update([])
        lost = [e for e in events if e["type"] == "lost"]
        assert len(lost) == 1

    def test_new_track_created_per_unmatched_det(self):
        tracker = PlateTracker(confirm_frames=3)
        # Two detections far apart → two tracks
        det1 = (0,   0,  100, 50, 0.9, "KL07BB1234")
        det2 = (400, 0,  500, 50, 0.9, "MH12AB3456")
        tracker.update([det1, det2])
        assert len(tracker.active_tracks) == 2

    def test_plate_consensus_majority_vote(self):
        """If 2/3 votes are for one plate, it wins."""
        tracker = PlateTracker(confirm_frames=3, vote_thresh=0.5)
        tracker.update([DET("KL07BB1234")])
        tracker.update([DET("KL07BB1234")])
        events = tracker.update([DET("KL07XX9999")])  # one wrong read
        confirmed = [e for e in events if e["type"] == "confirmed"]
        assert confirmed[0]["plate"] == "KL07BB1234"

    def test_reset_clears_state(self):
        tracker = PlateTracker(confirm_frames=2)
        tracker.update([DET("MH12AB3456")])
        tracker.reset()
        assert len(tracker.active_tracks) == 0
        # After reset, same plate should be confirmable again
        events = []
        for _ in range(2):
            events += tracker.update([DET("MH12AB3456")])
        assert any(e["type"] == "confirmed" for e in events)
