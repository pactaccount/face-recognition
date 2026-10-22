import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recognition"))
from matching import choose_identity


class MatchingTest(unittest.TestCase):
    def test_known_and_unknown(self):
        self.assertEqual(choose_identity(["A", "B"], [0.62, 0.31]), ("B", 0.31))
        self.assertEqual(choose_identity(["A"], [0.72]), ("Unknown", 0.72))


if __name__ == "__main__":
    unittest.main()
