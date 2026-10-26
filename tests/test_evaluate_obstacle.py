"""CPU-only tests for class-agnostic matching; no model or GPU is loaded."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_obstacle import match_boxes, score_counts


def gt(box, cls=0):
    return {"box": box, "class_id": cls}


def pred(box, confidence=0.9, cls=3):
    return {"box": box, "confidence": confidence, "class_id": cls}


class MatchTests(unittest.TestCase):
    def test_different_classes_still_match(self):
        matches, fp, fn = match_boxes([gt([0, 0, 10, 10], 0)], [pred([0, 0, 10, 10], cls=3)])
        self.assertEqual((len(matches), fp, fn), (1, [], []))

    def test_duplicate_prediction_is_false_positive(self):
        matches, fp, fn = match_boxes([gt([0, 0, 10, 10])],
                                      [pred([0, 0, 10, 10], .9), pred([0, 0, 10, 10], .8)])
        self.assertEqual((len(matches), fp, fn), (1, [1], []))

    def test_adjacent_objects_both_match_without_nms(self):
        truth = [gt([0, 0, 10, 10]), gt([8, 0, 18, 10], 2)]
        found = [pred([0, 0, 10, 10], .8), pred([8, 0, 18, 10], .9)]
        matches, fp, fn = match_boxes(truth, found)
        self.assertEqual((len(matches), fp, fn), (2, [], []))

    def test_no_predictions(self):
        self.assertEqual(match_boxes([gt([0, 0, 10, 10])], []), ([], [], [0]))

    def test_no_ground_truth(self):
        self.assertEqual(match_boxes([], [pred([0, 0, 10, 10])]), ([], [0], []))

    def test_empty_result(self):
        self.assertEqual(match_boxes([], []), ([], [], []))
        self.assertEqual(score_counts(0, 0, 0), {"precision": 0.0, "recall": None, "f1": 0.0})

    def test_confidence_order_is_used(self):
        matches, fp, fn = match_boxes([gt([0, 0, 10, 10])],
                                      [pred([0, 0, 10, 10], .1), pred([0, 0, 10, 10], .9)])
        self.assertEqual(matches[0]["prediction_index"], 1)
        self.assertEqual((fp, fn), ([0], []))


if __name__ == "__main__":
    unittest.main()
