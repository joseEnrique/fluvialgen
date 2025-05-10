import unittest
import pandas as pd
from datetime import datetime
from fluvialgen.past_forecast_batcher import PastForecastBatcher

class MockDataset:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item

    def take(self, n):
        """Return an iterator that yields n items from the dataset."""
        class TakeIterator:
            def __init__(self, data, n):
                self.data = data
                self.n = n
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= self.n or self.index >= len(self.data):
                    raise StopIteration
                item = self.data[self.index]
                self.index += 1
                return item

        return TakeIterator(self.data, n)

class TestPastForecastBatcher(unittest.TestCase):
    def setUp(self):
        self.data = [
            ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
            ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2),
            ({'moment': datetime(2024, 1, 1, 0, 2), 'value': 3}, 3),
            ({'moment': datetime(2024, 1, 1, 0, 3), 'value': 4}, 4),
            ({'moment': datetime(2024, 1, 1, 0, 4), 'value': 5}, 5),
            ({'moment': datetime(2024, 1, 1, 0, 5), 'value': 6}, 6)
        ]
        self.dataset = MockDataset(self.data)

    def tearDown(self):
        if hasattr(self, 'batcher'):
            self.batcher.stop()

    def test_initialization(self):
        """Test initialization of PastForecastBatcher"""
        batcher = PastForecastBatcher(self.dataset, past_size=3, forecast_size=1)
        
        self.assertEqual(batcher.past_size, 3)
        self.assertEqual(batcher.forecast_size, 1)
        self.assertEqual(batcher.buffer, [])
        self.assertEqual(batcher._count, 0)
        self.assertIsNone(batcher._last_element)

    def test_forecast_size_zero(self):
        """Test PastForecastBatcher with forecast_size=0"""
        self.batcher = PastForecastBatcher(self.dataset, past_size=2, forecast_size=0)
        
        # Get first instance
        X_past, y_past, current_x = self.batcher.get_message()
        
        # Check past data
        self.assertIsInstance(X_past, pd.DataFrame)
        self.assertIsInstance(y_past, pd.Series)
        self.assertEqual(len(X_past), 2)
        self.assertEqual(len(y_past), 2)
        self.assertEqual(X_past.iloc[0]['value'], 1)
        self.assertEqual(X_past.iloc[1]['value'], 2)
        self.assertEqual(y_past.iloc[0], 1)
        self.assertEqual(y_past.iloc[1], 2)
        
        # Check current x (should be the third element)
        self.assertEqual(current_x['value'], 3)
        self.assertEqual(current_x['moment'], datetime(2024, 1, 1, 0, 2))

    def test_forecast_size_one(self):
        """Test PastForecastBatcher with forecast_size=1"""
        self.batcher = PastForecastBatcher(self.dataset, past_size=2, forecast_size=1)
        
        # Get first instance
        X_past, y_past, current_x = self.batcher.get_message()
        
        # Check past data
        self.assertIsInstance(X_past, pd.DataFrame)
        self.assertIsInstance(y_past, pd.Series)
        self.assertEqual(len(X_past), 2)
        self.assertEqual(len(y_past), 2)
        self.assertEqual(X_past.iloc[0]['value'], 1)
        self.assertEqual(X_past.iloc[1]['value'], 2)
        self.assertEqual(y_past.iloc[0], 1)
        self.assertEqual(y_past.iloc[1], 2)
        
        # Check current x (should be the fourth element)
        self.assertEqual(current_x['value'], 4)
        self.assertEqual(current_x['moment'], datetime(2024, 1, 1, 0, 3))

    def test_forecast_size_two(self):
        """Test PastForecastBatcher with forecast_size=2"""
        self.batcher = PastForecastBatcher(self.dataset, past_size=2, forecast_size=2)
        
        # Get first instance
        X_past, y_past, current_x = self.batcher.get_message()
        
        # Check past data
        self.assertIsInstance(X_past, pd.DataFrame)
        self.assertIsInstance(y_past, pd.Series)
        self.assertEqual(len(X_past), 2)
        self.assertEqual(len(y_past), 2)
        self.assertEqual(X_past.iloc[0]['value'], 1)
        self.assertEqual(X_past.iloc[1]['value'], 2)
        self.assertEqual(y_past.iloc[0], 1)
        self.assertEqual(y_past.iloc[1], 2)
        
        # Check current x (should be the fifth element)
        self.assertEqual(current_x['value'], 5)
        self.assertEqual(current_x['moment'], datetime(2024, 1, 1, 0, 4))

    def test_insufficient_data(self):
        """Test PastForecastBatcher with insufficient data"""
        insufficient_data = [
            ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
            ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2)
        ]
        dataset = MockDataset(insufficient_data)
        self.batcher = PastForecastBatcher(dataset, past_size=1, forecast_size=1, n_instances=2)
        
        # Should raise StopIteration because we need at least past_size + 1 + forecast_size elements
        with self.assertRaises(StopIteration):
            self.batcher.get_message()



if __name__ == '__main__':
    unittest.main() 