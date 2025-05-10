import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from fluvialgen.movingwindow_generator import MovingWindowBatcher

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

class TestMovingWindowBatcher(unittest.TestCase):
    def setUp(self):
        """
        Initial setup for each test
        """
        self.data = [
            ({'moment': datetime(2024, 1, 1, 0, 0), 'value': 1}, 1),
            ({'moment': datetime(2024, 1, 1, 0, 1), 'value': 2}, 2),
            ({'moment': datetime(2024, 1, 1, 0, 2), 'value': 3}, 3),
            ({'moment': datetime(2024, 1, 1, 0, 3), 'value': 4}, 4),
            ({'moment': datetime(2024, 1, 1, 0, 4), 'value': 5}, 5),
            ({'moment': datetime(2024, 1, 1, 0, 5), 'value': 6}, 6)
        ]
        self.dataset = MockDataset(self.data)
        self.instance_size = 3
        self.batch_size = 2

    def test_window_creation(self):
        """
        Test that verifies the correct creation of sliding windows
        """
        batcher = MovingWindowBatcher(
            dataset=self.dataset,
            instance_size=self.instance_size,
            batch_size=self.batch_size,
            n_instances=10
        )

        # Get the first batch
        X, y = batcher.get_message()

        # Verify that X has the correct shape (batch_size * instance_size rows)
        self.assertEqual(X.shape[0], self.batch_size * self.instance_size)
        
        # Verify that y has the correct length (batch_size * instance_size)
        self.assertEqual(len(y), self.batch_size * self.instance_size)

        # Verify the content of the first window
        self.assertEqual(X.iloc[0]['value'], 1)
        self.assertEqual(X.iloc[1]['value'], 2)
        self.assertEqual(X.iloc[2]['value'], 3)
        self.assertEqual(y.iloc[0], 1)
        self.assertEqual(y.iloc[1], 2)
        self.assertEqual(y.iloc[2], 3)

        # Verify the content of the second window
        self.assertEqual(X.iloc[3]['value'], 1)
        self.assertEqual(X.iloc[4]['value'], 2)
        self.assertEqual(X.iloc[5]['value'], 3)
        self.assertEqual(y.iloc[3], 1)
        self.assertEqual(y.iloc[4], 2)
        self.assertEqual(y.iloc[5], 3)

        # Get the second batch
        X, y = batcher.get_message()

        # Verify the content of the first window in second batch
        self.assertEqual(X.iloc[0]['value'], 2)
        self.assertEqual(X.iloc[1]['value'], 3)
        self.assertEqual(X.iloc[2]['value'], 4)
        self.assertEqual(y.iloc[0], 2)
        self.assertEqual(y.iloc[1], 3)
        self.assertEqual(y.iloc[2], 4)

        # Verify the content of the second window in second batch
        self.assertEqual(X.iloc[3]['value'], 2)
        self.assertEqual(X.iloc[4]['value'], 3)
        self.assertEqual(X.iloc[5]['value'], 4)
        self.assertEqual(y.iloc[3], 2)
        self.assertEqual(y.iloc[4], 3)
        self.assertEqual(y.iloc[5], 4)

if __name__ == '__main__':
    unittest.main() 