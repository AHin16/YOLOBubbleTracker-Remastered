import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from demo_yolo11 import Args, MotionAnalyzer, build_tracker, process_frame, render_mask_frame
from visualization.visualize import plot_tracking


class TrackingContinuityTest(unittest.TestCase):
    @patch("demo_yolo11.render_mask_frame", return_value=None)
    def test_empty_frames_advance_both_trackers(self, _render):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        predictor = SimpleNamespace(predict=lambda image, timer: (None, {
            "raw_img": image, "height": 80, "width": 100,
        }))
        for tracker_type in ("botsort", "bytetrack"):
            with self.subTest(tracker=tracker_type):
                args = Args()
                args.tracker_type = tracker_type
                args.cmc_method = "none"
                tracker = build_tracker(args)
                timer = SimpleNamespace(stop=lambda average: .01)
                process_frame(frame, predictor, tracker, timer, args, 0, [])
                inner = tracker.tracker if tracker_type == "botsort" else tracker
                self.assertEqual(inner.frame_id, 1)

    def test_bytetrack_accepts_empty_array_and_reconnects_same_id(self):
        args = Args()
        args.tracker_type = "bytetrack"
        tracker = build_tracker(args)
        detection = torch.tensor([[20, 20, 50, 50, .95, 0]])
        initial = tracker.update(detection, [80, 100])
        self.assertEqual(len(initial), 1)
        self.assertEqual(len(tracker.update(np.empty((0, 6), np.float32), [80, 100])), 0)
        restored = tracker.update(detection, [80, 100])
        self.assertEqual(restored[0].track_id, initial[0].track_id)

    def test_turn_splits_segment_preserves_history_and_recovers(self):
        analyzer = MotionAnalyzer(SimpleNamespace(
            min_track_frames=2, frame_interval_seconds=1,
            max_turn_angle_deg=120, turn_min_segment_px=3,
        ))
        points = [(20, 20), (30, 20), (20, 20), (10, 20), (0, 20)]
        records = []
        for frame_id, center in enumerate(points, 1):
            record = {"track_id": 1, "center": center, "area_px": 100,
                      "passes_edge_filter": True}
            analyzer.update(frame_id, [record])
            records.append(record)
        self.assertFalse(records[2]["is_csv_sample"])
        self.assertTrue(records[3]["is_csv_sample"])
        self.assertTrue(records[4]["is_csv_sample"])
        self.assertEqual([row["frame"] for row in analyzer.csv_rows], [2, 4, 5])
        self.assertEqual(records[3]["trajectory"], [(20., 20.), (10., 20.)])

    @patch("demo_yolo11.plot_tracking", return_value=None)
    def test_edge_and_immature_observations_remain_drawable(self, draw):
        tracks = [{"track_id": 1, "tlwh": np.array([0, 0, 20, 20]),
                   "score": .9, "polygon": None, "trajectory": [(10, 10)],
                   "center": (10, 10), "speed_real_per_s": 0,
                   "is_drawable": True, "is_csv_sample": False, "is_visible": False}]
        render_mask_frame(np.zeros((40, 40, 3), np.uint8), tracks, 0, 0, Args())
        self.assertEqual(draw.call_args.args[2], [1])
        self.assertEqual(draw.call_args.kwargs["speeds"], [None])

    @patch("visualization.visualize.cv2.putText")
    def test_unmeasured_speed_is_not_labeled_as_zero(self, put_text):
        plot_tracking(np.zeros((100, 100, 3), np.uint8), [[20, 20, 30, 30]], [7],
                      centers=[(35, 35)], speeds=[None], show_speed_text=True, mode="mask")
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("ID 7", labels)
        self.assertFalse(any("ID 7  0.00" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
