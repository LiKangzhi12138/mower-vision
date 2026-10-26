"""CPU-only checks for explicit four-to-one label conversion."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_obstacle_dataset import mapped_label


class ConversionTests(unittest.TestCase):
    def test_coordinates_and_line_endings_unchanged(self):
        original = "2 0.123456 0.5 0.1 0.2\r\n3 0.8 0.9 0.02 0.03\n"
        mapped, count = mapped_label(original)
        self.assertEqual(mapped, "0 0.123456 0.5 0.1 0.2\r\n0 0.8 0.9 0.02 0.03\n")
        self.assertEqual(count, 2)

    def test_empty_negative_stays_empty(self):
        self.assertEqual(mapped_label(""), ("", 0))

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError):
            mapped_label("4 0.5 0.5 0.2 0.2\n")


if __name__ == "__main__":
    unittest.main()
