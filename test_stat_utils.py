import unittest

from post_process_utils import find_closest_non_missing, impute_missing_values

class TestImputationFunctions(unittest.TestCase):

    def test_find_closest_non_missing_left(self):
        self.assertEqual(find_closest_non_missing([1, -1, 3, -1], 2, "left"), (0, 1))
        self.assertEqual(find_closest_non_missing([1, -1, 3, -1], 1, "left"), (0, 1))
        self.assertEqual(find_closest_non_missing([-1, -1, 3, -1], 2, "left"), (None, None))
        self.assertEqual(find_closest_non_missing([-1, -1, -1, -1], 3, "left"), (None, None))
        
    def test_find_closest_non_missing_right(self):
        self.assertEqual(find_closest_non_missing([-1, -1, 3, -1], 1, "right"), (2, 3))
        self.assertEqual(find_closest_non_missing([-1, -1, -1, 4], 1, "right"), (3, 4))
        self.assertEqual(find_closest_non_missing([-1, -1, 3, -1], 2, "right"), (None, None))
        self.assertEqual(find_closest_non_missing([-1, -1, -1, -1], 0, "right"), (None, None))

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            find_closest_non_missing([-1, -1, 3, -1], 1, "middle")

    def test_impute_missing_values(self):
        self.assertEqual(impute_missing_values([1, -1, 3, -1]), [1, 2, 3, 3])
        self.assertEqual(impute_missing_values([1, -1, -1, 4]), [1, 2, 3, 4])
        self.assertEqual(impute_missing_values([-1, 2, -1, 4]), [2, 2, 3, 4])
        self.assertEqual(impute_missing_values([-1, -1, -1, 4]), [4, 4, 4, 4])
        self.assertEqual(impute_missing_values([-1, -1, 3, -1]), [3, 3, 3, 3])

if __name__ == '__main__':
    unittest.main()
