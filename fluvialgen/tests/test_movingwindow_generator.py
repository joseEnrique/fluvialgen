import unittest
import tracemalloc
from river import datasets
from fluvialgen.movingwindow_generator import MovingWindowBatcher

# Start tracemalloc
tracemalloc.start()

class TestMovingWindowBatcher(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.Bikes()
        self.batcher = MovingWindowBatcher(
            dataset=self.dataset,
            instance_size=2,
            batch_size=2,
            n_instances=10
        )

    def tearDown(self):
        if hasattr(self, 'batcher'):
            self.batcher.stop()
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'close'):
            self.dataset.close()
        # Get the current memory snapshot
        snapshot = tracemalloc.take_snapshot()
        # Display top 10 memory allocations
        top_stats = snapshot.statistics('lineno')
        print("\nMemory usage by line:")
        for stat in top_stats[:10]:
            print(stat)

    def test_batch_generation(self):
        try:
            # Test that we can get at least one batch
            X, y = next(self.batcher)
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertEqual(len(X), 2)  # batch_size
            self.assertEqual(len(y), 4)  # batch_size * instance_size
        finally:
            self.batcher.stop()

    def test_multiple_batches(self):
        try:
            # Test that we can get multiple batches
            batches = []
            for X, y in self.batcher:
                batches.append((X, y))
            self.assertGreater(len(batches), 0)
        finally:
            self.batcher.stop()

if __name__ == '__main__':
    unittest.main() 