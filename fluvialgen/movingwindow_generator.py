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

    def create_instance(self, start_idx):
        """
        Creates an instance (window) starting from start_idx.
        """
        if start_idx + self.instance_size > len(self.data_list):
            return None
        
        return self.data_list[start_idx:start_idx + self.instance_size]

    def _convert_to_pandas(self, batch):
        """
        Converts a batch of instances into a DataFrame (X) and a Series (y).
        
        Args:
            batch: List of instances, where each instance is a list of tuples (x,y)
            
        Returns:
            tuple: (pd.DataFrame, pd.Series) where:
            - X is a DataFrame with all features concatenated horizontally
            - y is a Series with all targets concatenated horizontally
        """
        X_rows = []
        y_rows = []
        
        for instance in batch:
            # For each instance, concatenate the data and targets horizontally
            x_row = []
            y_row = []
            
            for x, y in instance:
                # Convert x to a list if it is a dictionary (River format)
                if isinstance(x, dict):
                    x = list(x.values())
                x_row.extend(x)  # Concatenate features horizontally
                y_row.append(y)  # Concatenate targets horizontally
            
            X_rows.append(x_row)
            y_rows.extend(y_row)
            
        # Create DataFrame and Series
        X = pd.DataFrame(X_rows)
        y = pd.Series(y_rows)
        
        return X, y

    def get_message(self):
        """
        Obtains the next batch of instances and converts it to pandas format.
        Returns:
            tuple: (pd.DataFrame, pd.Series)
        Raises:
            StopIteration: When no more data is available
        """
        try:
            while True:
                # Get the next message from the River dataset
                x, y = super().get_message()
                
                # Add the message to the current window
                self.current_window.append((x, y))
                
                # If we have enough messages to form an instance
                if len(self.current_window) >= self.instance_size:
                    # Create an instance with the last instance_size messages
                    instance = self.current_window[-self.instance_size:]
                    self.current_batch.append(instance)
                    
                    # If we have enough instances for a batch
                    if len(self.current_batch) >= self.batch_size:
                        self._count += 1
                        batch = self.current_batch
                        self.current_batch = []  # Clear the current batch
                        return self._convert_to_pandas(batch)
                    
                # If there is no more data in the dataset
                if self._count >= self.n_instances:
                    raise StopIteration("No more data available")
                    
        except StopIteration:
            self.stop()
            raise

    def get_count(self):
        """
        Returns the total number of instances processed.
        """
        return self._count