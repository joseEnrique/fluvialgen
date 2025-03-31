import unittest
import numpy as np
import pandas as pd
from river import datasets

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from fluvialgen.movingwindow_generator import MovingWindowBatcher

class TestMovingWindowBatcher(unittest.TestCase):
    def setUp(self):
        """
        Initial setup for each test
        """
        self.dataset = datasets.Bikes()
        self.instance_size = 3
        self.batch_size = 3

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

        # Verify that X has the correct shape (batch_size rows)
        self.assertEqual(X.shape[0], self.batch_size)
        
        # Verify that y has the correct length (batch_size * instance_size)
        self.assertEqual(len(y), self.batch_size * self.instance_size)


if __name__ == '__main__':
    unittest.main() 