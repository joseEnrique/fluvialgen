from typing import Iterator, Tuple, Dict, Any
import numpy as np

from fluvialgen.past_forecast_batcher import PastForecastBatcher

class SequentialDataset:
    """
    Dataset that generates numerical sequences for testing with River.
    
    This dataset generates x values from 0 to x_max and y values from y_start to y_start+x_max.
    It's useful for testing time series models like DeepForecasterWithHistory.
    
    Parameters
    ----------
    x_max : int
        Maximum value for x (will generate values from 0 to x_max)
    y_start : int
        Initial value for y (will generate values from y_start to y_start+x_max)
    feature_name : str
        Feature name for the x dictionary
    """
    
    def __init__(self, x_max: int = 100, y_start: int = 100, feature_name: str = "value"):
        self.x_max = x_max
        self.y_start = y_start
        self.feature_name = feature_name
        self.index = 0
        
        # Generate complete sequences
        self.x_values = list(range(x_max + 1))  # [0, 1, 2, ..., x_max]
        self.y_values = list(range(y_start, y_start + x_max + 1))  # [y_start, y_start+1, ..., y_start+x_max]
        
    def __iter__(self) -> Iterator[Tuple[Dict[str, float], float]]:
        """Returns an iterator that produces (x, y) pairs."""
        self.index = 0
        return self
    
    def __next__(self) -> Tuple[Dict[str, float], float]:
        """Returns the next (x, y) pair in River-compatible format."""
        if self.index >= len(self.x_values):
            raise StopIteration
            
        # River format: x is a dictionary, y is a scalar value
        x = {self.feature_name: self.x_values[self.index]}
        y = self.y_values[self.index]
        
        self.index += 1
        return x, y
    
    def take(self, n: int) -> Iterator[Tuple[Dict[str, float], float]]:
        """Returns an iterator that produces n (x, y) pairs."""
        class TakeIterator:
            def __init__(self, dataset, n):
                self.dataset = dataset
                self.n = min(n, len(dataset.x_values))
                self.index = 0
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.index >= self.n:
                    raise StopIteration
                    
                x = {self.dataset.feature_name: self.dataset.x_values[self.index]}
                y = self.dataset.y_values[self.index]
                
                self.index += 1
                return x, y
                
        return TakeIterator(self, n)
    
    def __len__(self) -> int:
        """Returns the total number of examples in the dataset."""
        return len(self.x_values)

dataset = SequentialDataset(x_max=100, y_start=100, feature_name="value")

batcher = PastForecastBatcher(
        dataset=dataset,
        past_size=30,
        forecast_size=1,
        n_instances=10000
    )



for i,(x, y) in enumerate(batcher):
    print(x)
    print(y)
    #print(current_x)
    #print(current_y)
    #exit(0)