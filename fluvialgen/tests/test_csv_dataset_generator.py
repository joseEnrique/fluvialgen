import os
import unittest
import tempfile
from datetime import datetime

import pandas as pd

from fluvialgen.csv_dataset_generator import CSVDatasetGenerator


def _write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return tmp.name


class TestCSVDatasetGenerator(unittest.TestCase):

    def test_basic_iteration_with_explicit_features(self):
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02"],
                "value": [1, 2, 3],
                "c1": [10.0, 11.0, 12.0],
                "c2": [0, 1, 0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            gen = CSVDatasetGenerator(
                filepath=csv_path,
                target_column="value",
                feature_columns=["moment", "c1", "c2"],
                parse_dates=["moment"],
            )

            # First item
            x, y = gen.get_message()
            self.assertIsInstance(x, dict)
            self.assertEqual(y, 1)
            self.assertIn("moment", x)
            self.assertIn("c1", x)
            self.assertIn("c2", x)
            # Parsed date should be a pandas Timestamp
            self.assertTrue(isinstance(x["moment"], pd.Timestamp))

            # Second and third items
            x2, y2 = gen.get_message()
            x3, y3 = gen.get_message()
            self.assertEqual(y2, 2)
            self.assertEqual(y3, 3)
            self.assertEqual(gen.get_count(), 3)

            # Exceed n_instances -> StopIteration
            with self.assertRaises(StopIteration):
                gen.get_message()
        finally:
            os.unlink(csv_path)

    def test_auto_feature_columns_excludes_target(self):
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01"],
                "value": [1, 2],
                "c1": [10.0, 11.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            gen = CSVDatasetGenerator(
                filepath=csv_path,
                target_column="value",
                feature_columns=None,
                parse_dates=["moment"],
            )

            x, y = gen.get_message()
            self.assertEqual(y, 1)
            self.assertIn("moment", x)
            self.assertIn("c1", x)
            self.assertNotIn("value", x)
        finally:
            os.unlink(csv_path)

    def test_multi_target_iteration(self):
        """Test CSVDatasetGenerator with multiple targets"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02"],
                "target1": [1, 2, 3],
                "target2": [10, 20, 30],
                "c1": [0.1, 0.2, 0.3],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            gen = CSVDatasetGenerator(
                filepath=csv_path,
                target_column=["target1", "target2"],
                feature_columns=["moment", "c1"],
                parse_dates=["moment"],
            )

            # First item
            x, y = gen.get_message()
            self.assertIsInstance(x, dict)
            self.assertIsInstance(y, dict)
            self.assertEqual(y["target1"], 1)
            self.assertEqual(y["target2"], 10)
            self.assertIn("moment", x)
            self.assertIn("c1", x)
            self.assertNotIn("target1", x)
            self.assertNotIn("target2", x)

            # Second and third items
            x2, y2 = gen.get_message()
            x3, y3 = gen.get_message()
            self.assertEqual(y2["target1"], 2)
            self.assertEqual(y2["target2"], 20)
            self.assertEqual(y3["target1"], 3)
            self.assertEqual(y3["target2"], 30)
            self.assertEqual(gen.get_count(), 3)

            # Should raise StopIteration when exhausted
            with self.assertRaises(StopIteration):
                gen.get_message()
        finally:
            os.unlink(csv_path)

    def test_missing_target_column_raises(self):
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00"],
                "c1": [10.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            with self.assertRaises(ValueError):
                CSVDatasetGenerator(
                    filepath=csv_path,
                    target_column="value",
                    feature_columns=["moment", "c1"],
                )
        finally:
            os.unlink(csv_path)

    def test_missing_feature_column_raises(self):
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00"],
                "value": [1],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            with self.assertRaises(ValueError):
                CSVDatasetGenerator(
                    filepath=csv_path,
                    target_column="value",
                    feature_columns=["moment", "c1"],  # c1 does not exist
                )
        finally:
            os.unlink(csv_path)


if __name__ == "__main__":
    unittest.main()


