import pandas as pd
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from fluvialgen.base_generator import BaseGenerator


class CSVDatasetGenerator(BaseGenerator):
    """
    Stream-like generator over a CSV file that yields (x, y) tuples.
    Uses the same Base (BaseGenerator) timing/iteration behavior as other generators.
    Supports both single-target and multi-target scenarios.

    Parameters
    ----------
    filepath: str
        Path to the CSV file.
    target_column: Union[str, Sequence[str]]
        Column name(s) to use as target(s). Can be a string for single target
        or a list/sequence of strings for multi-target.
    feature_columns: Optional[Sequence[str]]
        Subset of columns to use as features X (excluding targets). If None, all
        columns except the targets will be used.
    parse_dates: Optional[Sequence[str]]
        Column names to parse as dates.
    stream_period: int
        Delay between consecutive messages (ms).
    timeout: int
        Maximum wait time (ms). Included for API completeness.
    """

    def __init__(
        self,
        filepath: str,
        target_column,  # Can be str or Sequence[str] for multi-target
        feature_columns: Optional[Sequence[str]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        stream_period: int = 0,
        timeout: int = 30000,
        **kwargs
    ):
        super().__init__(stream_period=stream_period, timeout=timeout)
        self.filepath = filepath
        
        # Handle both single target and multi-target
        if isinstance(target_column, str):
            self.target_columns = [target_column]
            self.is_multi_target = False
        else:
            self.target_columns = list(target_column)
            self.is_multi_target = True
        
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.parse_dates = list(parse_dates) if parse_dates is not None else None

        df = pd.read_csv(self.filepath, parse_dates=self.parse_dates)

        # Validate target columns
        for col in self.target_columns:
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in CSV")

        if self.feature_columns is None:
            self.feature_columns = [c for c in df.columns if c not in self.target_columns]
        else:
            for col in self.feature_columns:
                if col not in df.columns:
                    raise ValueError(f"Feature column '{col}' not found in CSV")

        # Build in-memory list of (x, y) where y can be single value or dict
        self._data: List[Tuple[dict, Union[float, dict]]] = []
        for _, row in df.iterrows():
            x = {col: row[col] for col in self.feature_columns}
            if self.is_multi_target:
                y = {col: row[col] for col in self.target_columns}
            else:
                y = row[self.target_columns[0]]
            self._data.append((x, y))

        self._iterator = iter(self._data)

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

