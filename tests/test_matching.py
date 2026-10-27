import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recognition"))
from matching import choose_identity, compute_confidence


class MatchingTest(unittest.TestCase):
    def test_known_and_unknown(self):
        self.assertEqual(choose_identity(["A", "B"], [0.62, 0.31]), ("B", 0.31))
        self.assertEqual(choose_identity(["A"], [0.72]), ("Unknown", 0.72))

    def test_nan_distance(self):
        self.assertEqual(choose_identity(["A", "B"], [float("nan"), 0.31]), ("B", 0.31))

    def test_confidence(self):
        self.assertEqual(compute_confidence(0.0, 0.55), 1.0)
        self.assertEqual(compute_confidence(0.55, 0.55), 0.0)
        self.assertEqual(compute_confidence(0.70, 0.55), 0.0)
        self.assertAlmostEqual(compute_confidence(0.275, 0.55), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
