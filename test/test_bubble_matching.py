import unittest

import numpy as np

from demo_yolo11 import Args, build_tracker


class BubbleMatchingTest(unittest.TestCase):
    def tracker(self):
        args = Args()
        args.fps = 172.1
        args.match_thresh = .7
        args.cmc_method = "none"
        return build_tracker(args)

    def test_overlapping_fast_bubbles_keep_their_shape_identity(self):
        tracker = self.tracker()
        frame = np.zeros((1016, 1264, 3), np.uint8)
        first = np.array([
            [462.77, 549.51, 800.15, 814.45, .958, 0],
            [342.68, 365.27, 701.45, 773.99, .854, 0],
        ], np.float32)
        second = np.array([
            [420.22, 437.11, 751.13, 681.26, .965, 0],
            [245.26, 227.42, 639.62, 611.30, .899, 0],
        ], np.float32)
        initial = tracker.update(first, frame)
        ids = [track.track_id for track in initial]
        targets = tracker.update(second, frame)
        assigned = {track.track_id: track.score for track in targets}
        self.assertAlmostEqual(assigned[ids[0]], .965, places=3)
        self.assertAlmostEqual(assigned[ids[1]], .899, places=3)

    def test_short_gap_uses_observed_velocity_and_old_ids_expire(self):
        tracker = self.tracker()
        frame = np.zeros((500, 500, 3), np.uint8)
        def detection(y):
            return np.array([[100, y, 180, y+60, .95, 0]], np.float32)
        initial = tracker.update(detection(300), frame)[0].track_id
        tracker.update(detection(260), frame)
        tracker.update(np.empty((0, 6), np.float32), frame)
        restored = tracker.update(detection(180), frame)
        self.assertEqual(restored[0].track_id, initial)
        for _ in range(6):
            tracker.update(np.empty((0, 6), np.float32), frame)
        new = tracker.update(detection(300), frame)
        self.assertTrue(all(track.track_id != initial for track in new))

    def test_bubble_remains_identifiable_while_leaving_top_edge(self):
        tracker = self.tracker()
        frame = np.zeros((500, 500, 3), np.uint8)
        first_id = None
        for top, bottom in [(140, 340), (60, 260), (0, 180), (0, 100), (0, 20)]:
            detection = np.array([[100, top, 250, bottom, .95, 0]], np.float32)
            targets = tracker.update(detection, frame)
            if first_id is None:
                first_id = targets[0].track_id
            self.assertEqual([target.track_id for target in targets], [first_id])

    def test_official_botsort_remains_available(self):
        args = Args()
        args.bubble_association = False
        args.cmc_method = "none"
        tracker = build_tracker(args)
        self.assertEqual(type(tracker.tracker).__name__, "BoTSORT")

    def test_shape_gate_still_applies_with_permissive_match_threshold(self):
        args = Args()
        args.match_thresh = 1.0
        tracker = build_tracker(args)
        frame = np.zeros((500, 500, 3), np.uint8)
        first = tracker.update(np.array([[100, 100, 200, 200, .95, 0]], np.float32), frame)
        first_id = first[0].track_id
        changed = tracker.update(np.array([[0, 0, 500, 500, .95, 0]], np.float32), frame)
        self.assertTrue(all(track.track_id != first_id for track in changed))


if __name__ == "__main__":
    unittest.main()
