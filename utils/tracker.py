"""
utils/tracker.py — Plate tracker for video streams.

Prevents duplicate entries when the same plate is visible across
multiple frames. Uses IoU-based box matching to link detections
across frames, and emits a "confirmed" event only after a plate
is seen N times consistently.

Usage:
    tracker = PlateTracker(confirm_frames=3, max_lost=10)

    for frame in frames:
        dets = model(frame)  # [(x1,y1,x2,y2,conf,plate_text), ...]
        events = tracker.update(dets)
        for event in events:
            if event["type"] == "confirmed":
                print(f"New plate: {event['plate']}")
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Track:
    track_id: int
    plate: Optional[str]
    bbox: Tuple[int, int, int, int]
    conf: float
    seen: int = 1
    lost: int = 0
    confirmed: bool = False
    votes: dict = field(default_factory=lambda: defaultdict(int))


def iou(a: tuple, b: tuple) -> float:
    """Compute Intersection over Union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class PlateTracker:
    """
    Simple IoU-based multi-object tracker for plate detections.

    Args:
        iou_thresh:      Minimum IoU to match a detection to an existing track.
        confirm_frames:  Frames a plate must be seen before emitting "confirmed".
        max_lost:        Frames without detection before a track is deleted.
        vote_thresh:     Minimum vote fraction for plate text consensus.
    """

    def __init__(
        self,
        iou_thresh: float = 0.35,
        confirm_frames: int = 3,
        max_lost: int = 10,
        vote_thresh: float = 0.5,
    ):
        self.iou_thresh = iou_thresh
        self.confirm_frames = confirm_frames
        self.max_lost = max_lost
        self.vote_thresh = vote_thresh
        self._tracks: List[Track] = []
        self._next_id = 0
        self._emitted: set = set()   # track_ids that have been confirmed

    def update(
        self,
        detections: List[Tuple],
    ) -> List[dict]:
        """
        Update tracker with new detections from current frame.

        detections: list of (x1, y1, x2, y2, conf, plate_text_or_None)

        Returns list of event dicts:
          {"type": "confirmed", "track_id": int, "plate": str, "conf": float}
          {"type": "lost",      "track_id": int, "plate": str}
        """
        events = []

        # --- Match detections to existing tracks (greedy IoU) ---
        matched_tracks = set()
        matched_dets = set()

        for det_i, det in enumerate(detections):
            x1, y1, x2, y2, conf, plate = det
            best_iou = self.iou_thresh
            best_track = None

            for track in self._tracks:
                if track.track_id in matched_tracks:
                    continue
                score = iou((x1, y1, x2, y2), track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track is not None:
                # Update matched track
                best_track.bbox = (x1, y1, x2, y2)
                best_track.conf = max(best_track.conf, conf)
                best_track.lost = 0
                best_track.seen += 1
                if plate:
                    best_track.votes[plate] += 1
                matched_tracks.add(best_track.track_id)
                matched_dets.add(det_i)
            # else: unmatched detection → new track (handled below)

        # --- Create new tracks for unmatched detections ---
        for det_i, det in enumerate(detections):
            if det_i in matched_dets:
                continue
            x1, y1, x2, y2, conf, plate = det
            t = Track(
                track_id=self._next_id,
                plate=plate,
                bbox=(x1, y1, x2, y2),
                conf=conf,
            )
            if plate:
                t.votes[plate] += 1
            self._tracks.append(t)
            self._next_id += 1

        # --- Age unmatched tracks ---
        for track in self._tracks:
            if track.track_id not in matched_tracks:
                track.lost += 1

        # --- Emit confirmed events ---
        for track in self._tracks:
            if (
                track.seen >= self.confirm_frames
                and track.track_id not in self._emitted
            ):
                # Consensus plate text from votes
                best_plate = self._consensus_plate(track)
                if best_plate:
                    events.append({
                        "type": "confirmed",
                        "track_id": track.track_id,
                        "plate": best_plate,
                        "conf": track.conf,
                        "bbox": track.bbox,
                    })
                    track.plate = best_plate
                    track.confirmed = True
                    self._emitted.add(track.track_id)

        # --- Emit lost events and prune dead tracks ---
        alive = []
        for track in self._tracks:
            if track.lost >= self.max_lost:
                if track.confirmed:
                    events.append({
                        "type": "lost",
                        "track_id": track.track_id,
                        "plate": track.plate,
                    })
            else:
                alive.append(track)
        self._tracks = alive

        return events

    def _consensus_plate(self, track: Track) -> Optional[str]:
        """Return the most-voted plate text if it meets the vote threshold."""
        if not track.votes:
            return None
        total = sum(track.votes.values())
        best = max(track.votes, key=track.votes.get)
        if track.votes[best] / total >= self.vote_thresh:
            return best
        return None

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.lost == 0]

    def reset(self):
        self._tracks.clear()
        self._emitted.clear()
        self._next_id = 0
