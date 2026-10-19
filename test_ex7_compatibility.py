"""Import smoke tests for the temporary Exercise 7 compatibility layer."""

import unittest

import ex3
import ex7_support


EX7_PUBLIC_NAMES = (
    "q5_ex7",
    "trying_estimate_projection_matrices_with_ransac_ex7",
    "get_sucees_estimation_ex7",
    "ex7_plot_supporters_non_supporters_after_ransac",
)


class Exercise7CompatibilityImportTest(unittest.TestCase):
    def test_public_names_import_from_support_module(self):
        for name in EX7_PUBLIC_NAMES:
            self.assertTrue(callable(getattr(ex7_support, name)))

    def test_ex3_exports_no_exercise7_specific_names(self):
        for name in EX7_PUBLIC_NAMES:
            self.assertFalse(hasattr(ex3, name))


if __name__ == "__main__":
    unittest.main()
