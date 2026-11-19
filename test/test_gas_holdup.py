#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import unittest

import numpy as np

from bubble_analyzer.gas_holdup import EdgeBubblePolicy, ROI, analyze_masks


class GasHoldupAnalysisTest(unittest.TestCase):
    def test_uses_union_of_segmentation_masks_for_area_and_ratio(self):
        first = np.zeros((10, 10), dtype=np.uint8)
        second = np.zeros((10, 10), dtype=np.uint8)
        first[1:5, 1:5] = 1
        second[3:7, 3:7] = 1

        result = analyze_masks(
            [first, second],
            frame_shape=(10, 10),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 2)
        self.assertEqual(result.bubble_area, 28)
        self.assertEqual(result.roi_area, 100)
        self.assertAlmostEqual(result.bubble_area_ratio, 28.0)
        expected_diameter = 2.0 * math.sqrt(16.0 / math.pi)
        self.assertAlmostEqual(result.average_diameter, expected_diameter)
        expected_volume = 2.0 * math.pi * expected_diameter ** 3 / 6.0
        self.assertAlmostEqual(result.gas_holdup, expected_volume)

    def test_sums_individual_spheres_with_calibration_and_view_depth(self):
        small = np.zeros((10, 10), dtype=np.uint8)
        large = np.zeros((10, 10), dtype=np.uint8)
        small[1:3, 1:3] = 1
        large[2:6, 2:6] = 1
        # With this calibration, areas 4 and 16 give actual diameters 2 and 4.
        scale = math.sqrt(math.pi) / 2.0
        for scale_factor, depth, expected_percent in ((1, 3, 16), (1, 6, 8), (2, 3, 32)):
            with self.subTest(scale_factor=scale_factor, depth=depth):
                result = analyze_masks(
                    [small, large], (10, 10), min_bubble_area=1,
                    real_units_per_pixel=scale * scale_factor, view_depth=depth,
                )
                self.assertEqual(result.bubble_area, 19)
                self.assertAlmostEqual(result.bubble_area_ratio, 19.0)
                self.assertAlmostEqual(result.bubble_volume, 12 * math.pi * scale_factor ** 3)
                self.assertAlmostEqual(result.roi_volume, 25 * math.pi * scale_factor ** 2 * depth)
                self.assertAlmostEqual(result.gas_holdup, expected_percent)

    def test_empty_frame_has_zero_bubble_volume_and_holdup(self):
        result = analyze_masks([], (10, 20), real_units_per_pixel=0.1, view_depth=2)
        self.assertEqual(result.bubble_count, 0)
        self.assertEqual(result.bubble_volume, 0.0)
        self.assertAlmostEqual(result.roi_volume, 4.0)
        self.assertEqual(result.gas_holdup, 0.0)

    def test_rejects_nonpositive_and_nonfinite_volume_parameters(self):
        for parameter in ("real_units_per_pixel", "view_depth"):
            for value in (0, -1, math.nan, math.inf):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        analyze_masks([], (10, 10), **{parameter: value})

    def test_applies_minimum_area_without_removing_valid_small_bubbles(self):
        noise = np.zeros((8, 8), dtype=np.uint8)
        bubble = np.zeros((8, 8), dtype=np.uint8)
        noise[1, 1] = 1
        bubble[2:4, 2:5] = 1

        result = analyze_masks(
            [noise, bubble],
            frame_shape=(8, 8),
            min_bubble_area=2,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 6)

    def test_clips_statistics_to_roi(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1

        result = analyze_masks(
            [mask],
            frame_shape=(10, 10),
            roi=ROI(x=5, y=5, width=5, height=5),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.INCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 9)
        self.assertEqual(result.roi_area, 25)
        self.assertAlmostEqual(result.bubble_area_ratio, 36.0)
        self.assertAlmostEqual(result.roi_volume, 25.0)
        self.assertAlmostEqual(result.gas_holdup, 100.0 * 36.0 / math.sqrt(math.pi) / 25.0)

    def test_exclude_ignores_masks_touching_the_image_border(self):
        edge_bubble = np.zeros((10, 10), dtype=np.uint8)
        inner_bubble = np.zeros((10, 10), dtype=np.uint8)
        edge_bubble[0:3, 2:5] = 1
        inner_bubble[4:7, 4:7] = 1

        result = analyze_masks(
            [edge_bubble, inner_bubble],
            frame_shape=(10, 10),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.EXCLUDE,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 9)

    def test_weighted_counts_only_visible_segmented_area(self):
        edge_bubble = np.zeros((6, 6), dtype=np.uint8)
        edge_bubble[0:2, 0:3] = 1

        result = analyze_masks(
            [edge_bubble],
            frame_shape=(6, 6),
            min_bubble_area=1,
            edge_policy=EdgeBubblePolicy.WEIGHTED,
        )

        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.bubble_area, 6)
        self.assertAlmostEqual(result.bubble_area_ratio, 100.0 * 6.0 / 36.0)
        volume = 4.0 * 6.0 ** 1.5 / (3.0 * math.sqrt(math.pi))
        self.assertAlmostEqual(result.gas_holdup, 100.0 * volume / 36.0)

    def test_rejects_masks_with_unexpected_dimensions(self):
        bad_mask = np.zeros((5, 5), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "mask shape"):
            analyze_masks(
                [bad_mask],
                frame_shape=(10, 10),
                min_bubble_area=1,
                edge_policy=EdgeBubblePolicy.INCLUDE,
            )


if __name__ == "__main__":
    unittest.main()
