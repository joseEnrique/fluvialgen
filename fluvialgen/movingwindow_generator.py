from fluvialgen.river_dataset_generator import RiverDatasetGenerator
import pandas as pd

class MovingWindowBatcher(RiverDatasetGenerator):
    """
    A generator for moving windows over a River dataset with batching support.
    """
    def __init__(
        self,
        dataset,
        instance_size: int,
        batch_size: int,
        stream_period: int = 0,
        timeout: int = 30000,
        n_instances: int = 1000,
        **kwargs
    ):
        """
        Args:
            dataset: The River dataset to iterate over
            instance_size: Size of each window
            batch_size: Number of instances per batch
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
        self.instance_size = instance_size
        self.batch_size = batch_size
        self.current_window = []
        self._count = 0
        self.current_batch = []
        self.buffer = []
        self.windows = []
        self.window_idx = 0

    def create_instance(self, start_idx):
        """
        Creates an instance (window) starting from start_idx.
        """
        if start_idx + self.instance_size > len(self.data_list):
            return None
        
        return self.data_list[start_idx:start_idx + self.instance_size]

    def _convert_to_pandas(self, batch):
        """
        Converts a batch of instances to a DataFrame (X) and a Series (y).
        
        Args:
            batch: List of instances, where each instance is a list of tuples (x,y)
            
        Returns:
            tuple: (pd.DataFrame, pd.Series) where:
            - X is a DataFrame with all x data from all instances
            - y is a Series with all y values from all instances
        """
        all_x_data = []  # List for all x values
        all_y_data = []  # List for all y values
        
        # For each instance in the batch
        for instance in batch:
            # For each tuple (x,y) in the instance
            for x, y in instance:
                # If x is a dictionary, convert it to a list
                if isinstance(x, dict):
                    x = list(x.values())
                all_x_data.append(x)  # Add x to the list
                all_y_data.append(y)  # Add y to the list
        
        # Create DataFrame and Series
        X = pd.DataFrame(all_x_data)
        y = pd.Series(all_y_data)
        
        return X, y

    def get_message(self):
        """
        Gets the next batch of instances with sliding windows in pandas format.
        Always returns exactly batch_size instances, with overlap between consecutive instances.
        For example, with instance_size=2, the pattern would be:
        Batch 1: [x1,x2], [x2,x3] -> DataFrame with [x1,x2,x2,x3], Series with [y1,y2,y2,y3]
        Batch 2: [x2,x3], [x3,x4] -> DataFrame with [x2,x3,x3,x4], Series with [y2,y3,y3,y4]
        
        If there's not enough data for a full batch, the last elements will be duplicated
        to ensure we always return exactly batch_size * instance_size elements.
        """
        try:
            # We need at least instance_size elements to form an instance
            required_min_elements = self.instance_size
            
            # Attempt to fill the buffer
            while len(self.buffer) < required_min_elements and self._count < self.n_instances:
                try:
                    x, y = super().get_message()
                    self.buffer.append((x, y))
                except StopIteration:
                    # If we can't get more data, break the loop
                    break
            
            # If we can't form even one instance, stop iteration
            if len(self.buffer) < self.instance_size:
                raise StopIteration("No more data available")
            
            # Create a batch with exactly batch_size instances
            batch = []
            
            # Calculate how many instances we can create from the buffer
            available_instances = max(1, len(self.buffer) - self.instance_size + 1)
            
            # Create up to batch_size instances
            for i in range(min(self.batch_size, available_instances)):
                instance = self.buffer[i:i+self.instance_size]
                batch.append(instance)
            
            # If we don't have enough instances to fill a batch, duplicate the last instance
            while len(batch) < self.batch_size:
                # Duplicate the last instance
                batch.append(batch[-1])
            
            # Update buffer by removing the first element (sliding window)
            if len(self.buffer) > 0:
                self.buffer.pop(0)
            
            self._count += 1
            return self._convert_to_pandas(batch)
            
        except StopIteration:
            self.stop()
            raise

    def get_count(self):
        """
        Returns the total number of instances processed.
        """
        return self._count