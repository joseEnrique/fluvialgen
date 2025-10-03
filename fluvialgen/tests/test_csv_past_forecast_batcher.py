import os
import unittest
import tempfile
from datetime import datetime

import pandas as pd

from fluvialgen.csv_past_forecast_batcher import CSVPastForecastBatcher


def _write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return tmp.name


class TestCSVPastForecastBatcher(unittest.TestCase):

    def test_forecast_size_zero(self):
        """Test CSVPastForecastBatcher with forecast_size=0"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03", "2024-01-01 00:04", "2024-01-01 00:05"],
                "value": [1, 2, 3, 4, 5, 6],
                "c1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "c2": [0, 1, 0, 1, 0, 1],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            batcher = CSVPastForecastBatcher(
                filepath=csv_path,
                target_column="value",
                feature_columns=["moment", "c1", "c2"],
                parse_dates=["moment"],
                past_size=2,
                forecast_size=0,
                n_instances=2
            )
            
            # Get first instance
            X_past, y_past = batcher.get_message()
            
            # Check past data
            self.assertIsInstance(X_past, pd.DataFrame)
            self.assertEqual(len(X_past), 2)  # past_size=2
            self.assertEqual(X_past.iloc[0]['c1'], 10.0)
            self.assertEqual(X_past.iloc[1]['c1'], 11.0)
            
            # Check forecast value (y_past is now a single value)
            self.assertEqual(y_past, 3)  # Element at past_size + forecast_size (2+0=2)
            
            # Get second instance
            X_past2, y_past2 = batcher.get_message()
            self.assertEqual(len(X_past2), 2)
            self.assertEqual(X_past2.iloc[0]['c1'], 11.0)
            self.assertEqual(X_past2.iloc[1]['c1'], 12.0)
            self.assertEqual(y_past2, 4)  # Element at position 3
            
            self.assertEqual(batcher.get_count(), 2)
        finally:
            os.unlink(csv_path)

    def test_forecast_size_one(self):
        """Test CSVPastForecastBatcher with forecast_size=1"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03", "2024-01-01 00:04"],
                "value": [1, 2, 3, 4, 5],
                "c1": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            batcher = CSVPastForecastBatcher(
                filepath=csv_path,
                target_column="value",
                feature_columns=["moment", "c1"],
                parse_dates=["moment"],
                past_size=2,
                forecast_size=1,
                n_instances=2
            )
            
            # Get first instance
            X_past, y_past = batcher.get_message()
            
            # Check past data
            self.assertIsInstance(X_past, pd.DataFrame)
            self.assertEqual(len(X_past), 2)  # past_size=2
            self.assertEqual(X_past.iloc[0]['c1'], 10.0)
            self.assertEqual(X_past.iloc[1]['c1'], 11.0)
            
            # Check forecast value (y_past is now a single value)
            self.assertEqual(y_past, 4)  # Element at past_size + forecast_size (2+1=3)
            
            self.assertEqual(batcher.get_count(), 1)
        finally:
            os.unlink(csv_path)

    def test_auto_feature_columns_excludes_target(self):
        """Test that feature_columns=None excludes target column"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03"],
                "value": [1, 2, 3, 4],
                "c1": [10.0, 11.0, 12.0, 13.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            batcher = CSVPastForecastBatcher(
                filepath=csv_path,
                target_column="value",
                feature_columns=None,  # Auto-select features
                parse_dates=["moment"],
                past_size=2,
                forecast_size=0,
                n_instances=2
            )
            
            X_past, y_past = batcher.get_message()
            self.assertEqual(y_past, 3)
            self.assertIn("moment", X_past.columns)
            self.assertIn("c1", X_past.columns)
            self.assertNotIn("value", X_past.columns)
        finally:
            os.unlink(csv_path)

    def test_insufficient_data_raises(self):
        """Test CSVPastForecastBatcher with insufficient data"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00", "2024-01-01 00:01"],
                "value": [1, 2],
                "c1": [10.0, 11.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            batcher = CSVPastForecastBatcher(
                filepath=csv_path,
                target_column="value",
                feature_columns=["moment", "c1"],
                parse_dates=["moment"],
                past_size=1,
                forecast_size=1,
                n_instances=2
            )
            
            # Should raise StopIteration because we need at least past_size + 1 + forecast_size elements
            with self.assertRaises(StopIteration):
                batcher.get_message()
        finally:
            os.unlink(csv_path)

    def test_missing_target_column_raises(self):
        """Test error when target column is missing"""
        df = pd.DataFrame(
            {
                "moment": ["2024-01-01 00:00"],
                "c1": [10.0],
            }
        )
        csv_path = _write_temp_csv(df)
        try:
            with self.assertRaises(ValueError):
                CSVPastForecastBatcher(
                    filepath=csv_path,
                    target_column="value",
                    feature_columns=["moment", "c1"],
                )
        finally:
            os.unlink(csv_path)


if __name__ == "__main__":
    unittest.main()
