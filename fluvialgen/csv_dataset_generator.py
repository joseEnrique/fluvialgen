import pandas as pd
from typing import Iterator, List, Optional, Sequence, Tuple

from fluvialgen.base_generator import BaseGenerator


class CSVDatasetGenerator(BaseGenerator):
    """
    Stream-like generator over a CSV file that yields (x, y) tuples.
    Uses the same Base (BaseGenerator) timing/iteration behavior as other generators.

    Parameters
    ----------
    filepath: str
        Path to the CSV file.
    target_column: str
        Column name to use as the target y.
    feature_columns: Optional[Sequence[str]]
        Subset of columns to use as features X (excluding target). If None, all
        columns except the target will be used.
    parse_dates: Optional[Sequence[str]]
        Column names to parse as dates.
    stream_period: int
        Delay between consecutive messages (ms).
    timeout: int
        Maximum wait time (ms). Included for API completeness.
    n_instances: int
        Maximum number of instances to iterate over.
    """

    def __init__(
        self,
        filepath: str,
        target_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        stream_period: int = 0,
        timeout: int = 30000,
        n_instances: int = 1000,
        **kwargs
    ):
        super().__init__(stream_period=stream_period, timeout=timeout)
        self.filepath = filepath
        self.target_column = target_column
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.parse_dates = list(parse_dates) if parse_dates is not None else None
        self.n_instances = n_instances

        df = pd.read_csv(self.filepath, parse_dates=self.parse_dates)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in CSV")

        if self.feature_columns is None:
            self.feature_columns = [c for c in df.columns if c != self.target_column]
        else:
            for col in self.feature_columns:
                if col not in df.columns:
                    raise ValueError(f"Feature column '{col}' not found in CSV")

        # Build in-memory list of (x, y) then create an iterator limited by n_instances
        self._data: List[Tuple[dict, float]] = []
        for _, row in df.iterrows():
            x = {col: row[col] for col in self.feature_columns}
            y = row[self.target_column]
            self._data.append((x, y))

        self._iterator = iter(self._data[: self.n_instances])

    def __next__(self):
        # Respect BaseGenerator timing logic
        super().__next__()
        return self.get_message()

    def get_message(self):
        try:
            x, y = next(self._iterator)
            self._count += 1
            return x, y
        except StopIteration:
            self.stop()
            raise

    def get_count(self):
        return self._count

    def stop(self):
        # Nothing to close explicitly; set iterator to None for GC symmetry
        if hasattr(self, "_iterator"):
            self._iterator = None

