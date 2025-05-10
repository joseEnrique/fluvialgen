import unittest
import pandas as pd
from river import datasets
from fluvialgen.past_forecast_batcher import PastForecastBatcher

class TestPastForecastBatcher(unittest.TestCase):
    def setUp(self):
        """
        Initial setup for each test
        """
        self.dataset = datasets.Bikes()
        self.past_size = 3
        self.forecast_size = 0  # Changed to 0 to test current moment

    def test_window_creation(self):
        """
        Test that verifies the correct creation of past and forecast windows
        """
        batcher = PastForecastBatcher(
            dataset=self.dataset,
            past_size=self.past_size,
            forecast_size=self.forecast_size,
            n_instances=10
        )

        # Get the first instance
        X_past, y_past, current_x = batcher.get_message()

        # Verify that X_past has the correct shape
        self.assertEqual(X_past.shape[0], self.past_size)
        
        # Verify that y_past has the correct length
        self.assertEqual(len(y_past), self.past_size)
        
        # Verify that current_x is a dict (the raw x value)
        self.assertIsInstance(current_x, dict)

    def test_multiple_instances(self):
        """
        Test that verifies we can get multiple instances
        """
        batcher = PastForecastBatcher(
            dataset=self.dataset,
            past_size=self.past_size,
            forecast_size=self.forecast_size,
            n_instances=10
        )

        # Get multiple instances
        instances = []
        for X_past, y_past, current_x in batcher:
            instances.append((X_past, y_past, current_x))
            if len(instances) >= 3:  # Limit to 3 instances for testing
                break

        # Verify we got at least one instance
        self.assertGreater(len(instances), 0, "No instances were generated")

        # Verify each instance has the correct shape
        for X_past, y_past, current_x in instances:
            self.assertEqual(X_past.shape[0], self.past_size)
            self.assertEqual(len(y_past), self.past_size)
            self.assertIsInstance(current_x, dict)

if __name__ == '__main__':
    unittest.main() 