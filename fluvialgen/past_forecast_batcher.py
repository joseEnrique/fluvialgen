from fluvialgen.river_dataset_generator import RiverDatasetGenerator
import pandas as pd

class PastForecastBatcher(RiverDatasetGenerator):
    """
    A generator for creating instances with past data and forecast windows.
    current_x and current_y always represent the current moment, including the last element.
    """
    def __init__(
        self,
        dataset,
        past_size: int,
        forecast_size: int = 0,
        stream_period: int = 0,
        timeout: int = 30000,
        n_instances: int = 1000,
        **kwargs
    ):
        """
        Args:
            dataset: The River dataset to iterate over
            past_size: Number of past instances to include in each window
            forecast_size: Number of future instances to include in each window (not used for current_x/y)
            stream_period: Delay between consecutive messages (ms)
            timeout: Maximum wait time (ms)
            n_instances: Maximum number of instances to process
        """
        super().__init__(
            dataset=dataset,
            stream_period=stream_period,
            timeout=timeout,
            n_instances=n_instances,
            **kwargs
        )
        self.past_size = past_size
        self.forecast_size = forecast_size
        self.buffer = []
        self._count = 0
        self._last_element = None

    def _convert_to_pandas(self, instance):
        """
        Converts an instance to pandas objects.
        
        Args:
            instance: A tuple (past_x, past_y, current_x, current_y)
            
        Returns:
            tuple: (pd.DataFrame, pd.Series, dict, float) where:
            - X_past is a DataFrame with all past x data
            - y_past is a Series with all past y values
            - current_x is the current x value (dict)
            - current_y is the current y value (float)
        """
        past_x, past_y, current_x, current_y = instance
        
        # Create DataFrame and Series
        X_past = pd.DataFrame(past_x)
        y_past = pd.Series(past_y)
        
        return X_past, y_past, current_x, current_y

    def get_message(self):
        """
        Gets the next instance with past and forecast windows in pandas format.
        current_x and current_y represent the elements at current position + forecast_size.
        
        For example, with past_size=3 and forecast_size=1:
        At time t=4:
        - X_past = DataFrame with [x1, x2, x3]
        - y_past = Series with [y1, y2, y3]
        - current_x = x5 (current moment + forecast_size)
        - current_y = y5 (current moment + forecast_size)
        """
        try:
            # We need at least past_size + 1 + forecast_size elements to form an instance
            required_min_elements = self.past_size + 1 + self.forecast_size
            
            # Try to get the next element
            try:
                x, y = super().get_message()
                self.buffer.append((x, y))
                self._last_element = (x, y)
            except StopIteration:
                # If we can't get more data and we don't have enough elements, stop
                if len(self.buffer) < required_min_elements:
                    raise StopIteration("No more data available")
            
            # If we don't have enough elements yet, try to get more
            while len(self.buffer) < required_min_elements and self._count < self.n_instances:
                try:
                    x, y = super().get_message()
                    self.buffer.append((x, y))
                    self._last_element = (x, y)
                except StopIteration:
                    # If we can't get more data and we don't have enough elements, stop
                    if len(self.buffer) < required_min_elements:
                        raise StopIteration("No more data available")
                    break
            
            # Get past data
            past_data = self.buffer[:self.past_size]
            past_x = [x for x, _ in past_data]
            past_y = [y for _, y in past_data]
            
            # Get current x and y (the elements at past_size + forecast_size)
            if len(self.buffer) <= self.past_size + self.forecast_size:
                raise StopIteration("Not enough data for current_x/y")
            current_x = self.buffer[self.past_size + self.forecast_size][0]
            current_y = self.buffer[self.past_size + self.forecast_size][1]
            
            # Create instance
            instance = (past_x, past_y, current_x, current_y)
            
            # Remove the first element if we have more elements
            if len(self.buffer) > self.past_size:
                self.buffer.pop(0)
            
            self._count += 1
            return self._convert_to_pandas(instance)
            
        except StopIteration:
            self.stop()
            raise

    def get_count(self):
        """
        Returns the total number of instances processed.
        """
        return self._count 